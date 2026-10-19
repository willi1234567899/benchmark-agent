@delete
Feature: Tessera purge benchmark (POST /tessera/v1/purge)
  In order to evaluate delete performance characteristics
  As a test engineer
  I want to run purge benchmarks against the Tessera time-series store on existing data

  Background:
    Given a SUT Tessera endpoint is configured and reachable
    And the target store from the SUT config is available

  Scenario: purge benchmark (smoke+load by default, whole-series purge with -D purge_all=true)
    When I run the purge benchmark
    Then I store the purge benchmark result as "reports/purge.json"
    And I store the purge benchmark context as "reports/purge-context.json"

  Scenario Outline: purge a series over a time range
    When I purge series "<series>" from "<from>" until "<until>" with label filter "<label_filter>"
    Then the purge is acknowledged
    And a fetch of series "<series>" over the same range returns 0 rows

    Examples:
      | series                | from                 | until                | label_filter |
      | bddbench_single_write | 1970-01-01T00:00:00Z | 2100-01-01T00:00:00Z | none         |
      | bddbench_single_write | 2026-08-26T00:00:00Z | 2026-08-27T00:00:00Z | none         |
      | bddbench_single_write | 1970-01-01T00:00:00Z | 2100-01-01T00:00:00Z | host=web-01  |
