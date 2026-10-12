//! Planning-time statistics measured on a sample of a table's row groups.
//!
//! A cost model needs two numbers a Parquet footer does not carry: how many
//! rows a scan's filters keep, and how many distinct values a column holds.
//! Both are read off a handful of row groups spread evenly over the table and
//! scaled to its size: a filter's selectivity on the sample, and a distinct
//! count extrapolated from how often the sample repeats a value.

use std::sync::Arc;

use ahash::RandomState;
use arrow_arith::boolean::and_kleene;
use arrow_array::cast::AsArray;
use arrow_array::{Array, BooleanArray, RecordBatch};
use arrow_row::{RowConverter, SortField};
use dispatch::{DataFlowDispatcher, Projection};
use planner::compile::{ExprEvalFn, ExprFn};
use planner::expression::Expression;

use crate::reading::table_input;
use crate::types::table::ParquetTable;

/// How many rows a sample aims to cover. Whole row groups are read, so the
/// sample is the fewest evenly spaced row groups that reach this many rows (or
/// every row group of a smaller table, which makes the measurement exact).
const SAMPLE_ROWS: i64 = 1 << 20;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("compiling a filter for sampling: {0}")]
    Compile(#[from] planner::compile::Error),
    #[error("scanning the sample: {0}")]
    Scan(#[from] dispatch::DataFlowError),
    #[error("row-encoding column {column} for a distinct count: {source}")]
    RowEncoding {
        column: usize,
        source: arrow_schema::ArrowError,
    },
}

/// Estimate how many rows of `table` satisfy every filter in `filters`, by
/// measuring their selectivity on a sample of its row groups. The filters
/// reference the table's columns by storage index; nulls do not pass. A table
/// with rows never estimates below one row, since a sample that caught no
/// match does not prove there is none.
pub fn estimate_filtered_row_count(
    dispatcher: &DataFlowDispatcher,
    table: &ParquetTable,
    filters: &[Expression],
) -> Result<u64, Error> {
    let total_rows = table.total_rows() as u64;
    if filters.is_empty() || total_rows == 0 {
        return Ok(total_rows);
    }

    let mut columns = Vec::new();
    for filter in filters {
        filter.collect_column_refs(&mut columns);
    }
    columns.sort_unstable();
    columns.dedup();
    // The sample batch holds only the referenced columns, in `columns` order.
    let mut filters: Vec<Expression> = filters.to_vec();
    for filter in &mut filters {
        filter.visit_column_refs_mut(&mut |reference| {
            reference.column_idx = columns
                .binary_search(&reference.column_idx)
                .expect("every referenced column is in the sample projection");
        });
    }
    let compiled: Vec<ExprFn> = filters
        .iter()
        .map(Expression::compile)
        .collect::<Result<_, _>>()?;
    let compiled = Arc::new(compiled);

    let sample = sample_of(table);
    let counts: Vec<(u64, u64)> =
        table_input(dispatcher, &sample, Projection::columns(columns), false)
            .map(move || {
                let mut evaluators: Vec<ExprEvalFn> =
                    compiled.iter().map(|build| build()).collect();
                move |batch: RecordBatch| count_matching_rows(&batch, &mut evaluators)
            })
            .execute()
            .collect()?;
    let (sampled_rows, matching_rows) = counts.iter().fold(
        (0u64, 0u64),
        |(rows, matching), &(batch_rows, batch_matching)| {
            (rows + batch_rows, matching + batch_matching)
        },
    );
    if sampled_rows == 0 {
        return Ok(total_rows);
    }
    let estimate = (total_rows as u128 * matching_rows as u128 / sampled_rows as u128) as u64;
    Ok(estimate.max(1))
}

/// The batch's row count and how many of its rows every evaluator keeps.
fn count_matching_rows(batch: &RecordBatch, evaluators: &mut [ExprEvalFn]) -> (u64, u64) {
    let rows = batch.num_rows();
    let mut kept: Option<BooleanArray> = None;
    for evaluate in evaluators.iter_mut() {
        let mask = evaluate(batch).into_array(rows);
        let mask = mask.as_boolean();
        kept = Some(match kept {
            None => mask.clone(),
            Some(kept) => {
                and_kleene(&kept, mask).expect("boolean masks of one batch share its length")
            }
        });
    }
    let matching = kept.map_or(rows, |kept| kept.true_count());
    (rows as u64, matching as u64)
}

/// Estimate how many distinct non-null values `column` of `table` holds, from
/// a sample of its row groups. Values are counted per sampled batch, so a
/// column whose equal values sit together (a sorted key, say) is read as it
/// is: the sample shows no value in more than one batch, bar the ones cut by
/// a batch boundary, and its distinct count scales with the table. A column
/// whose values recur across batches is extrapolated from how many values one
/// batch holds against how many two do (Chao's estimator), which an all-batch
/// value, as in a column of a few values, leaves at what the sample saw. The
/// estimate never exceeds the table's row count.
pub fn estimate_distinct_count(
    dispatcher: &DataFlowDispatcher,
    table: &ParquetTable,
    column: usize,
) -> Result<u64, Error> {
    let total_rows = table.total_rows() as u64;
    if total_rows == 0 {
        return Ok(0);
    }
    let sample = sample_of(table);
    let hashed: Vec<Result<Vec<u64>, Error>> =
        table_input(dispatcher, &sample, Projection::columns([column]), false)
            .map(move || {
                let hasher = RandomState::with_seeds(0x5eed, 0x1234, 0xabcd, 0x9876);
                let mut converter: Option<RowConverter> = None;
                move |batch: RecordBatch| {
                    hash_column_values(&batch, column, &mut converter, &hasher)
                }
            })
            .execute()
            .collect()?;
    let sampled_rows = sample.total_rows() as u64;
    let mut batches = 0u64;
    // One entry per (value, batch) pair: each batch's values, deduplicated.
    let mut occurrences = Vec::new();
    for batch_hashes in hashed {
        let mut batch_hashes = batch_hashes?;
        batches += u64::from(!batch_hashes.is_empty());
        batch_hashes.sort_unstable();
        batch_hashes.dedup();
        occurrences.extend(batch_hashes);
    }
    occurrences.sort_unstable();

    let mut distinct = 0u64;
    let mut in_one_batch = 0u64;
    let mut in_two_batches = 0u64;
    let mut run = 0u64;
    for i in 0..occurrences.len() {
        run += 1;
        if i + 1 == occurrences.len() || occurrences[i + 1] != occurrences[i] {
            distinct += 1;
            in_one_batch += u64::from(run == 1);
            in_two_batches += u64::from(run == 2);
            run = 0;
        }
    }
    if sampled_rows >= total_rows {
        return Ok(distinct);
    }
    let shared_values = distinct - in_one_batch;
    let values_sit_together = shared_values < batches && 2 * in_one_batch >= distinct;
    if values_sit_together {
        return Ok((distinct * total_rows / sampled_rows).min(total_rows));
    }
    let unseen =
        in_one_batch.saturating_mul(in_one_batch.saturating_sub(1)) / (2 * (in_two_batches + 1));
    Ok(distinct.saturating_add(unseen).min(total_rows))
}

/// Hash every non-null value of the batch's `column` through its row encoding,
/// building the converter from the first batch's column type.
fn hash_column_values(
    batch: &RecordBatch,
    column: usize,
    converter: &mut Option<RowConverter>,
    hasher: &RandomState,
) -> Result<Vec<u64>, Error> {
    let values = batch.column(0);
    let row_encoding = |source| Error::RowEncoding { column, source };
    if converter.is_none() {
        *converter = Some(
            RowConverter::new(vec![SortField::new(values.data_type().clone())])
                .map_err(row_encoding)?,
        );
    }
    let rows = converter
        .as_mut()
        .expect("the converter was just built")
        .convert_columns(std::slice::from_ref(values))
        .map_err(row_encoding)?;
    Ok((0..values.len())
        .filter(|&i| values.is_valid(i))
        .map(|i| hasher.hash_one(rows.row(i).as_ref()))
        .collect())
}

/// The evenly spaced row groups of `table` that make up its sample, as a table
/// of their own.
fn sample_of(table: &ParquetTable) -> Arc<ParquetTable> {
    let row_groups = table.row_groups();
    let total_rows = table.total_rows();
    let rows_per_group = (total_rows / row_groups.len() as i64).max(1);
    let sample_groups = (SAMPLE_ROWS / rows_per_group).clamp(1, row_groups.len() as i64) as usize;
    let mut sample = table.clone();
    *sample.row_groups_mut() = (0..sample_groups)
        .map(|i| row_groups[i * row_groups.len() / sample_groups].clone())
        .collect();
    Arc::new(sample)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{ArrayRef, Int32Array, Scalar, StringViewArray};
    use arrow_schema::{DataType, Field, Schema};
    use dispatch::Dispatch;
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;
    use planner::expression::{Compare, CompareType, Ref};
    use planner::types::Type;
    use std::fs::File;
    use std::sync::OnceLock;
    use tempfile::TempDir;

    /// A shared one-worker dispatch pool for this module's tests.
    fn test_dispatcher() -> &'static DataFlowDispatcher {
        static DISPATCH: OnceLock<Dispatch> = OnceLock::new();
        DISPATCH
            .get_or_init(|| Dispatch::spin_up(1, 32, None))
            .dispatcher()
    }

    /// `batch` written as one file whose row groups hold `rows_per_group` rows.
    fn write_table(batch: &RecordBatch, rows_per_group: usize) -> (TempDir, ParquetTable) {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("data.parquet");
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(rows_per_group))
            .build();
        let mut writer =
            ArrowWriter::try_new(File::create(&path).unwrap(), batch.schema(), Some(props))
                .unwrap();
        writer.write(batch).unwrap();
        writer.close().unwrap();
        let table = ParquetTable::from_files(test_dispatcher(), &[path], &[]).unwrap();
        (dir, table)
    }

    /// `(id, name)` rows for ids `0..rows`, named `n<id>`.
    fn id_name_batch(rows: i32) -> RecordBatch {
        let names: Vec<String> = (0..rows).map(|id| format!("n{id}")).collect();
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int32, false),
                Field::new("name", DataType::Utf8View, false),
            ])),
            vec![
                Arc::new(Int32Array::from_iter_values(0..rows)) as ArrayRef,
                Arc::new(StringViewArray::from_iter_values(&names)) as ArrayRef,
            ],
        )
        .unwrap()
    }

    fn compare(
        column: usize,
        column_type: Type,
        compare_type: CompareType,
        value: ArrayRef,
    ) -> Expression {
        Expression::Compare(Compare {
            left: Box::new(Expression::Ref(Ref {
                column_idx: column,
                return_type: column_type,
                name: None,
            })),
            right: Box::new(Expression::Constant(Scalar::new(value))),
            compare_type,
            return_type: Type::Boolean,
        })
    }

    fn id_at_least(value: i32) -> Expression {
        let value = Arc::new(Int32Array::from(vec![value])) as ArrayRef;
        compare(0, Type::Int32, CompareType::GreaterEqual, value)
    }

    fn name_is(value: &str) -> Expression {
        let value = Arc::new(StringViewArray::from(vec![value])) as ArrayRef;
        compare(1, Type::Utf8, CompareType::Equal, value)
    }

    #[test]
    fn a_filter_estimate_scales_the_table_by_its_sampled_selectivity() {
        let (_dir, table) = write_table(&id_name_batch(100), 25);

        let estimate = estimate_filtered_row_count(test_dispatcher(), &table, &[id_at_least(70)]);

        assert_eq!(estimate.unwrap(), 30);
    }

    #[test]
    fn filters_combine_and_a_later_column_is_read_as_itself() {
        let (_dir, table) = write_table(&id_name_batch(100), 25);

        let estimate = estimate_filtered_row_count(
            test_dispatcher(),
            &table,
            &[id_at_least(70), name_is("n71")],
        );

        assert_eq!(estimate.unwrap(), 1);
    }

    #[test]
    fn a_sample_without_a_match_still_estimates_one_row() {
        let (_dir, table) = write_table(&id_name_batch(100), 25);

        let estimate = estimate_filtered_row_count(test_dispatcher(), &table, &[id_at_least(1000)]);

        assert_eq!(estimate.unwrap(), 1);
    }

    #[test]
    fn no_filters_estimate_the_whole_table() {
        let (_dir, table) = write_table(&id_name_batch(100), 25);

        let estimate = estimate_filtered_row_count(test_dispatcher(), &table, &[]);

        assert_eq!(estimate.unwrap(), 100);
    }

    #[test]
    fn a_fully_sampled_column_counts_its_distinct_values_exactly() {
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("cycle", DataType::Int32, false),
                Field::new("unique", DataType::Int32, false),
            ])),
            vec![
                Arc::new(Int32Array::from_iter_values((0..100).map(|i| i % 10))) as ArrayRef,
                Arc::new(Int32Array::from_iter_values(0..100)) as ArrayRef,
            ],
        )
        .unwrap();
        let (_dir, table) = write_table(&batch, 25);

        let cycle = estimate_distinct_count(test_dispatcher(), &table, 0).unwrap();
        let unique = estimate_distinct_count(test_dispatcher(), &table, 1).unwrap();

        assert_eq!((cycle, unique), (10, 100));
    }

    #[test]
    fn nulls_are_not_distinct_values() {
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, true)])),
            vec![Arc::new(Int32Array::from(vec![Some(1), None, Some(1), None])) as ArrayRef],
        )
        .unwrap();
        let (_dir, table) = write_table(&batch, 4);

        let distinct = estimate_distinct_count(test_dispatcher(), &table, 0).unwrap();

        assert_eq!(distinct, 1);
    }

    /// Two columns over more rows than one sample covers: one cycling through
    /// a thousand values and one unique per row.
    fn large_table() -> (TempDir, ParquetTable) {
        let rows = 2 * SAMPLE_ROWS as i32;
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("cycle", DataType::Int32, false),
                Field::new("unique", DataType::Int32, false),
            ])),
            vec![
                Arc::new(Int32Array::from_iter_values((0..rows).map(|i| i % 1000))) as ArrayRef,
                Arc::new(Int32Array::from_iter_values(0..rows)) as ArrayRef,
            ],
        )
        .unwrap();
        write_table(&batch, 1 << 17)
    }

    #[test]
    fn a_sampled_repeating_column_counts_the_values_it_saw() {
        let (_dir, table) = large_table();

        let distinct = estimate_distinct_count(test_dispatcher(), &table, 0).unwrap();

        assert_eq!(distinct, 1000);
    }

    #[test]
    fn a_sampled_column_without_repeats_is_taken_as_unique() {
        let (_dir, table) = large_table();

        let distinct = estimate_distinct_count(test_dispatcher(), &table, 1).unwrap();

        assert_eq!(distinct, 2 * SAMPLE_ROWS as u64);
    }

    #[test]
    fn a_sampled_column_sorted_by_value_scales_what_it_saw() {
        let rows = 2 * SAMPLE_ROWS as i32;
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "sorted",
                DataType::Int32,
                false,
            )])),
            vec![Arc::new(Int32Array::from_iter_values((0..rows).map(|i| i / 4))) as ArrayRef],
        )
        .unwrap();
        let (_dir, table) = write_table(&batch, 1 << 17);

        let distinct = estimate_distinct_count(test_dispatcher(), &table, 0).unwrap();

        assert_eq!(distinct, (rows / 4) as u64);
    }

    #[test]
    fn a_sampled_filter_estimate_is_scaled_to_the_table() {
        let (_dir, table) = large_table();
        let half = Arc::new(Int32Array::from(vec![SAMPLE_ROWS as i32])) as ArrayRef;
        let filter = compare(1, Type::Int32, CompareType::GreaterEqual, half);

        let estimate = estimate_filtered_row_count(test_dispatcher(), &table, &[filter]).unwrap();

        assert_eq!(estimate, SAMPLE_ROWS as u64);
    }
}
