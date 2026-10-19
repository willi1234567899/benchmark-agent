Feature: InfluxDB gold live HTTP smoke

  Scenario: write, query, and delete the pinned batch
    Given InfluxDB connection settings are available
    When the pinned write batch is sent
    Then the write is accepted
    When the pinned read query is sent
    Then the query is accepted
    When the pinned delete is sent
    Then the delete is accepted
