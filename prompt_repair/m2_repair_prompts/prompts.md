# Frozen M2 prompts (verbatim)

`{expression}` is replaced by the referring expression. Decoding, image preprocessing,
and parsing are identical to the matched M1 runs of the same model.

## Baselines (each model's best fixed M1 interface)

**Qwen3-VL-4B — `tuple_box_norm1000`**

```
Locate the object described by the expression in the image. Return only one tuple in this exact format: (x1, y1, x2, y2). Coordinates must be normalized integers from 0 to 1000, where 0 is the image origin and 1000 is the image width or height. Do not include words, markdown, or explanation.

Expression: {expression}
```

**InternVL3.5-4B — `internvl_native_ref`**

```
Please provide the bounding box coordinate of the region this sentence describes: <ref>{expression}</ref>
```

## Relation-aware repair

**Qwen3-VL-4B — `relation_repair_tuple_norm1000`**

```
Locate the object described by the expression. Resolve every spatial and relational cue before choosing among similar objects, including left/right, front/behind, above/below, and relations to other objects. Return only one tight bounding-box tuple in this exact format: (x1, y1, x2, y2). Coordinates must be normalized integers from 0 to 1000. Do not include words, markdown, or explanation.

Expression: {expression}
```

**InternVL3.5-4B — `internvl_relation_repair`**

```
Resolve the spatial and relational cues in the sentence and distinguish the intended instance from similar objects. Then provide only the tight bounding box coordinate of the region this sentence describes: <ref>{expression}</ref>
```

## Part-aware repair

**Qwen3-VL-4B — `part_repair_tuple_norm1000`**

```
Locate exactly what the expression refers to. If it refers to a visible object part, return the smallest tight box enclosing that part rather than the whole object. If it refers to the whole object, return its tight box. Return only one tuple in this exact format: (x1, y1, x2, y2). Coordinates must be normalized integers from 0 to 1000. Do not include words, markdown, or explanation.

Expression: {expression}
```

**InternVL3.5-4B — `internvl_part_repair`**

```
If the sentence refers to an object part, provide the smallest tight bounding box of that part rather than the whole object; otherwise provide the tight whole-object box. Return only the bounding box coordinate for: <ref>{expression}</ref>
```
