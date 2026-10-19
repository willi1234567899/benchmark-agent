# Feedback Hints

Configured mode: `hybrid`

Automatic sources (runtime):

- validator feedback for malformed Gherkin or invalid Python
- execution feedback once the evaluation pipeline fails a run

Curated hint pool (scripted / hybrid), replayed deterministically by feedback round:

## Hint 1
Ensure the feature uses a Scenario Outline with Examples covering write, read, and delete.

## Hint 2
Keep step definitions `behave`-compatible: import from `behave`, and match step text exactly.

## Hint 3
Map operations to InfluxDB endpoints: write -> `/api/v2/write`, read -> `/api/v2/query`, delete -> `/api/v2/delete`.

## Hint 4
Return valid JSON only for the artifact schema (`feature`, `steps_py`), with no markdown fences.

## Hint 5
Separate preparation and execution steps where helpful, but keep the module importable and syntactically valid.
