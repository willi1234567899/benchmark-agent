@read @singlestore
Feature: Tessera fetch benchmark (POST /tessera/v1/fetch)
  In order to evaluate read performance characteristics
  As a test engineer
  I want to run fetch benchmarks against the Tessera time-series store with adjustable parameters

  Background:
    Given a SUT Tessera endpoint is configured and reachable
    And the target store from the SUT config is available

  Scenario Outline: generic fetch benchmark run
    When I run a generic fetch benchmark on series "<series>" with window "<window_from>" and result bound <max_rows> with <concurrent_clients> concurrent clients and label filter "<label_filter>"
    Then I store the generic fetch benchmark result as "reports/fetch-<id>.json"
    And I store the fetch benchmark context as "reports/fetch-context-<id>.json"

    @normal
    Examples:
      | id     | series                | window_from | max_rows | concurrent_clients | label_filter |
      | smoke  | bddbench_single_write | -1h         | 10       | 1                  | none         |
      | load   | bddbench_single_write | -6h         | 1000     | 4                  | none         |
      | stress | bddbench_single_write | -24h        | 10000    | 8                  | host         |
      | spike  | bddbench_single_write | -1h         | 100      | 32                 | none         |
      | soak   | bddbench_single_write | -1h         | 100      | 2                  | none         |
