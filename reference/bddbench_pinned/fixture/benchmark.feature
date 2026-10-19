@singlebucket
Feature: InfluxDB v2 write, query and delete benchmark on the pinned workload (bddbench port)
  In order to evaluate write, query and delete performance characteristics on the pinned workload
  As a test engineer
  I want to run the bddbench write, query and delete benchmarks against InfluxDB v2 with the pinned parameters

  Background:
    Given a SUT InfluxDB v2 endpoint is configured
    And the target bucket from the SUT config is set

  @write
  Scenario Outline: generic write benchmark run (batch + parallel)
    When I run a generic write benchmark on measurement "<measurement>" with batch size <batch_size>, <parallel_writers> parallel writers, compression "<compression>", timestamp precision "<precision>", point complexity "<point_complexity>", tag cardinality <tag_cardinality> and time ordering "<time_ordering>" for <batches> batches
    Then I store the generic write benchmark result as "reports/write-<id>.json"
    And I store the write benchmark context as "reports/write-context-<id>.json"

    @pinned
    Examples:
      | id     | measurement | batch_size | parallel_writers | compression | precision | point_complexity | tag_cardinality | time_ordering | batches |
      | pinned | benchmark   | 100        | 1                | none        | ms        | low              | 5               | in_order      | 10      |

  @query
  Scenario Outline: generic query benchmark run
    When I run a generic query benchmark on measurement "<measurement>" with time range "<time_range>" using query type "<query_type>" on tag host "<host>" and result size "<result_size>" with <concurrent_clients> concurrent clients, <repeats> sequential repeats, output format "<output_format>" and compression "<compression>"
    Then I store the generic query benchmark result as "reports/query-<id>.json"

    @pinned
    Examples:
      | id     | measurement | time_range | query_type | host | result_size | concurrent_clients | repeats | output_format | compression |
      | pinned | benchmark   | 2h         | filter     | h1   | small       | 1                  | 20      | csv           | none        |

  @delete
  Scenario Outline: delete benchmark (one delete by measurement and tag over the full time range)
    When I run the delete benchmark on measurement "<measurement>" for tag host "<host>"
    Then I store the generic delete benchmark result as "reports/delete-<id>.json"

    @pinned
    Examples:
      | id     | measurement | host |
      | pinned | benchmark   | h1   |
