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
Map operations to Tessera endpoints: write -> `POST /tessera/v1/ingest`, read -> `POST /tessera/v1/fetch`, delete -> `POST /tessera/v1/purge`. Every data request needs the API key in the `X-Tessera-Key` header and the `space` query parameter; bodies are JSON.

## Hint 4
Return valid JSON only for the artifact schema (`feature`, `steps_py`), with no markdown fences.

## Hint 5
Separate preparation and execution steps where helpful, but keep the module importable and syntactically valid.
