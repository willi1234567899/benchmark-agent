Feature: Kestrel gold HTTP smoke

  Scenario: write, read, and remove the pinned batch
    Given Kestrel connection settings are available
    When the pinned write batch is sent
    Then the write is accepted
    When the pinned read is sent
    Then the read is accepted
    When the pinned removal is sent
    Then the removal is accepted
