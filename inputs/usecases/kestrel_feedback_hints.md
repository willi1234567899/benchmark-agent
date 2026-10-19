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
Map operations to Kestrel verbs on the points resource `/v1/tenants/{tenant}/stores/{store}/series/{series}/points`: write -> `PUT` (columnar JSON body, `unit` query parameter), read -> `GET` (`since`, `limit`, `label.<name>`), delete -> `DELETE` (`from`, `to`, `label.<name>`). Every request needs the API token in the `X-Api-Token` header.

## Hint 4
Return valid JSON only for the artifact schema (`feature`, `steps_py`), with no markdown fences.

## Hint 5
Separate preparation and execution steps where helpful, but keep the module importable and syntactically valid.
