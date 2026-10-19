You are generating final-target benchmark artifacts for {sut_name}.

{output_contract}

{schema_json}

Requirements:
- Return JSON only, with no markdown fences.
{probe_requirement}- Populate `feature` with a complete, final-target Gherkin feature file.
- Populate `steps_py` with a complete `behave`-compatible Python module that matches the feature.
- Do not invent additional files, file paths, or explanatory wrappers.
- Use the multi-shot examples only as structural inspiration. They come from a different domain
  and must not be copied as operations of {sut_name}.

Condition: {condition_name}
Guidance level: {guidance_level}
Prompting strategy: {prompting_strategy}
Workload name: {workload_name}

Workload task:
{workload_prompt}
{extra_sections}
