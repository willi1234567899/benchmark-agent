@delete
Feature: Kestrel delete benchmark (DELETE .../series/{series}/points)
  In order to evaluate delete performance characteristics
  As a test engineer
  I want to run delete benchmarks against the Kestrel metrics service on existing data

  Background:
    Given a SUT Kestrel endpoint is configured and reachable
    And the target tenant and store from the SUT config are available

  Scenario: delete benchmark (smoke+load by default, whole-series removal with -D remove_all=true)
    When I run the delete benchmark
    Then I store the delete benchmark result as "reports/delete.json"
    And I store the delete benchmark context as "reports/delete-context.json"

  Scenario Outline: remove points of a series over a time range
    When I remove points of series "<series>" from "<from>" to "<to>" with label filter "<label_filter>"
    Then the removal is acknowledged
    And a read of series "<series>" over the same range returns 0 points

    Examples:
      | series                | from                 | to                   | label_filter |
      | bddbench_single_write | 1970-01-01T00:00:00Z | 2100-01-01T00:00:00Z | host=h1      |
      | bddbench_single_write | 2026-08-26T00:00:00Z | 2026-08-27T00:00:00Z | none         |
      | bddbench_single_write | 1970-01-01T00:00:00Z | 2100-01-01T00:00:00Z | none         |
