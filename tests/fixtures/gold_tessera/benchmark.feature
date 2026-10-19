Feature: Tessera gold HTTP smoke

  Scenario: ingest, fetch, and purge the pinned batch
    Given Tessera connection settings are available
    When the pinned ingest batch is sent
    Then the ingest is accepted
    When the pinned fetch is sent
    Then the fetch is accepted
    When the pinned purge is sent
    Then the purge is accepted
