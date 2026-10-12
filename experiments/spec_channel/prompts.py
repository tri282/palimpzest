"""Prompt templates for the Specification channel pilot (upstream spec writer, downstream executor, reference spec).

Design and results: REPORT.md.
"""

DI_FIELDS = ("name", "addr", "phone", "type")

DI_WRITER = """You are writing instructions for an assistant that will fill in the missing city value of restaurant records, one record at a time. The records come from a dataset with its own labeling conventions.

Below are example records with their correct city values. Write concise imputation rules, at most 250 words, that capture the conventions these examples imply: which city names to use, at what granularity, how to spell them, and which cues in the other fields decide the choice.

Output the rules only.

{examples}"""

DI_REF_SPEC = "Fill in the missing city for this restaurant record."

DI_EXECUTOR = """{spec}

Record: {record}

What is the missing city? Answer with the city name only."""


def di_record(r: dict) -> str:
    return "; ".join(f"{k}: {r.get(k) or '-'}" for k in DI_FIELDS)


def di_examples(rows: list[dict]) -> str:
    return "\n".join(f"{di_record(r)}; city: {r['city']}" for r in rows)


EM_FIELDS =("title", "category", "brand", "modelno", "price")

EM_WRITER = """You are writing instructions for an assistant that will decide, one pair at a time, whether a Walmart product record and an Amazon product record refer to the same real-world product.

Below are labeled example pairs. Write concise matching rules, at most 250 words, that capture the criteria these labels imply: how to treat model numbers, brands, product variants (color, size, capacity), bundles and accessories, and missing or noisy fields.

Output the rules only.

{examples}"""

EM_REF_SPEC = "Decide whether the two product records refer to the same real-world product."

EM_EXECUTOR = """{spec}

Record A (Walmart): {left}
Record B (Amazon): {right}

Do A and B refer to the same product? Answer Yes or No only."""


AB_FIELDS = ("name", "description", "price")

AB_WRITER = EM_WRITER.replace("a Walmart product record and an Amazon product record", "an Abt product record and a Buy product record")

AB_EXECUTOR = EM_EXECUTOR.replace("Record A (Walmart)", "Record A (Abt)").replace("Record B (Amazon)", "Record B (Buy)")


def em_record(r: dict, fields: tuple = EM_FIELDS) -> str:
    return "; ".join(f"{k}: {r.get(k) or '-'}" for k in fields)


def em_examples(pairs: list[dict], fields: tuple = EM_FIELDS) -> str:
    return "\n\n".join(f"Record A: {em_record(p['left'], fields)}\nRecord B: {em_record(p['right'], fields)}\nSame product: {'Yes' if p['label'] else 'No'}"
                       for p in pairs)
