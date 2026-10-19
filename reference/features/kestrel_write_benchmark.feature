@write @singlestore
Feature: Kestrel write benchmark (PUT .../series/{series}/points)
  In order to evaluate write performance characteristics
  As a test engineer
  I want to run write benchmarks against the Kestrel metrics service with adjustable parameters

  Background:
    Given a SUT Kestrel endpoint is configured and reachable
    And the target tenant and store from the SUT config are available

  Scenario Outline: generic write benchmark run (batch + parallel)
    When I run a generic write benchmark on series "<series>" with batch size <batch_size>, <parallel_writers> parallel writers, time unit "<unit>", point complexity "<point_complexity>", label cardinality <label_cardinality> and time ordering "<time_ordering>" for <batches> batches
    Then I store the generic write benchmark result as "reports/write-<id>.json"
    And I store the write benchmark context as "reports/write-context-<id>.json"

    @normal
    Examples:
      | id     | series                | batch_size | parallel_writers | unit | point_complexity | label_cardinality | time_ordering | batches |
      | smoke  | bddbench_single_write | 100        | 1                | ms   | low              | 5                 | in_order      | 10      |
      | load   | bddbench_single_write | 250        | 2                | ms   | medium           | 100               | in_order      | 10      |
      | stress | bddbench_single_write | 500        | 3                | ms   | high             | 250               | in_order      | 20      |
      | spike  | bddbench_single_write | 250        | 8                | ms   | medium           | 100               | in_order      | 3       |
      | soak   | bddbench_single_write | 250        | 2                | ms   | medium           | 100               | in_order      | 60      |

    @experimental
    Examples:
      | id         | series                | batch_size | parallel_writers | unit | point_complexity | label_cardinality | time_ordering | batches |
      | smoke      | bddbench_single_write | 250        | 2                | s    | medium           | 100               | out_of_order  | 5       |
      | load       | bddbench_single_write | 1000       | 4                | s    | high             | 1000              | out_of_order  | 50      |
      | breakpoint | bddbench_single_write | 500        | 32               | ns   | high             | 10000             | in_order      | 20      |
