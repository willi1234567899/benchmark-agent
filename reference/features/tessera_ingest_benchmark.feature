@write @singlestore
Feature: Tessera ingest benchmark (POST /tessera/v1/ingest)
  In order to evaluate write performance characteristics
  As a test engineer
  I want to run ingest benchmarks against the Tessera time-series store with adjustable parameters

  Background:
    Given a SUT Tessera endpoint is configured and reachable
    And the target store from the SUT config is available

  Scenario Outline: generic ingest benchmark run (batch + parallel)
    When I run a generic ingest benchmark on series "<series>" with batch size <batch_size>, <parallel_writers> parallel writers, timestamp unit "<ts_unit>", record complexity "<record_complexity>", label cardinality <label_cardinality> and time ordering "<time_ordering>" for <batches> batches
    Then I store the generic ingest benchmark result as "reports/ingest-<id>.json"
    And I store the ingest benchmark context as "reports/ingest-context-<id>.json"

    @normal
    Examples:
      | id     | series                | batch_size | parallel_writers | ts_unit | record_complexity | label_cardinality | time_ordering | batches |
      | smoke  | bddbench_single_write | 100        | 1                | ns      | low               | 10                | in_order      | 10      |
      | load   | bddbench_single_write | 250        | 2                | ns      | medium            | 100               | in_order      | 10      |
      | stress | bddbench_single_write | 500        | 3                | ns      | high              | 250               | in_order      | 20      |
      | spike  | bddbench_single_write | 250        | 8                | ns      | medium            | 100               | in_order      | 3       |
      | soak   | bddbench_single_write | 250        | 2                | ns      | medium            | 100               | in_order      | 60      |

    @experimental
    Examples:
      | id         | series                | batch_size | parallel_writers | ts_unit | record_complexity | label_cardinality | time_ordering | batches |
      | smoke      | bddbench_single_write | 250        | 2                | ms      | medium            | 100               | out_of_order  | 5       |
      | load       | bddbench_single_write | 1000       | 4                | ms      | high              | 1000              | out_of_order  | 50      |
      | stress     | bddbench_single_write | 1500       | 8                | ms      | high              | 5000              | out_of_order  | 50      |
      | spike      | bddbench_single_write | 250        | 32               | ms      | medium            | 200               | out_of_order  | 3       |
      | soak       | bddbench_single_write | 250        | 4                | ms      | medium            | 200               | out_of_order  | 200     |
      | breakpoint | bddbench_single_write | 500        | 32               | ns      | high              | 10000             | in_order      | 20      |
