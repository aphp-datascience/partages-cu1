import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from confit import Cli
from spacy.tokens import Doc, Span

import edsnlp
import edsnlp.data.converters
import edsnlp.pipes as eds
from edsnlp.data.converters import FILENAME


def clean_and_drop_ents(
    drop_ratio: float = 0.05,
    seed: Optional[int] = None,
    remove_pseudo_antipatterns: bool = False,
):
    rng = random.Random(seed)

    def strip_note_id_and_remove_some_ents(
        doc: Doc,
    ) -> Doc:
        """
        Transforms doc._.note_id to remove everything until the last "/" and remove the extension
        ex doc._.note_id == "folder/text.json" -> "text"

        And removes `drop_ratio` of entities at random (in doc.ents)

        Operates on a copy of the doc.

        seed:
          If provided, makes the entity-dropping deterministic.
        """
        new_doc = doc.copy()

        note_id = None
        try:
            note_id = doc._.note_id
        except AttributeError:
            pass

        if note_id:
            normalized = str(note_id).replace("\\", "/")
            basename = normalized.rsplit("/", 1)[-1]
            stem = os.path.splitext(basename)[0]
            print("note", note_id, "->", stem)
            try:
                new_doc._.note_id = stem
            except AttributeError:
                pass

        ents = list(doc.ents)
        if not ents:
            new_doc.ents = ()
            return new_doc

        assert 0 <= drop_ratio < 1
        kept = []
        for i, e in enumerate(ents):
            if rng.random() < drop_ratio:
                print("Randomly dropped", repr(e.text), "in", new_doc._.note_id)
                continue
            # CU1 patterns only
            if "\n" in e.text or remove_pseudo_antipatterns and (
                "date" in e.label_
                and (
                    "semaine" in e.text
                    or "mois" in e.text
                    or "jour" in e.text
                    or len(e.text) == 2
                    and e.text.startswith("J")
                    or "an" in e.text.split()
                    or "ans" in e.text.split()
                )
                or e.text == "IPP"
                or e.text.lower()
                in ("monsieur", "madame", "frère", "soeur", "père", "mère")
            ):
                print("Dropped", repr(e.text), "in", new_doc._.note_id)
                continue
            kept.append(e)
        new_doc.ents = [Span(new_doc, e.start, e.end, label=e.label) for e in kept]
        return new_doc

    return strip_note_id_and_remove_some_ents


BASE_TYPE_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "de.tudarmstadt.ukp.clarin.webanno.api.type.FeatureDefinition": {
        "%NAME": "de.tudarmstadt.ukp.clarin.webanno.api.type.FeatureDefinition",
        "%SUPER_TYPE": "uima.cas.TOP",
        "layer": {
            "%NAME": "layer",
            "%RANGE": "de.tudarmstadt.ukp.clarin.webanno.api.type.LayerDefinition",
        },
        "name": {"%NAME": "name", "%RANGE": "uima.cas.String"},
        "uiName": {"%NAME": "uiName", "%RANGE": "uima.cas.String"},
    },
    "de.tudarmstadt.ukp.clarin.webanno.api.type.LayerDefinition": {
        "%NAME": "de.tudarmstadt.ukp.clarin.webanno.api.type.LayerDefinition",
        "%SUPER_TYPE": "uima.cas.TOP",
        "name": {"%NAME": "name", "%RANGE": "uima.cas.String"},
        "uiName": {"%NAME": "uiName", "%RANGE": "uima.cas.String"},
    },
    "de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.morph.MorphologicalFeatures": {
        "%NAME": "de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.morph.MorphologicalFeatures",
        "%SUPER_TYPE": "uima.tcas.Annotation",
        "gender": {"%NAME": "gender", "%RANGE": "uima.cas.String"},
        "number": {"%NAME": "number", "%RANGE": "uima.cas.String"},
        "case": {"%NAME": "case", "%RANGE": "uima.cas.String"},
        "degree": {"%NAME": "degree", "%RANGE": "uima.cas.String"},
        "verbForm": {"%NAME": "verbForm", "%RANGE": "uima.cas.String"},
        "tense": {"%NAME": "tense", "%RANGE": "uima.cas.String"},
        "mood": {"%NAME": "mood", "%RANGE": "uima.cas.String"},
        "voice": {"%NAME": "voice", "%RANGE": "uima.cas.String"},
        "definiteness": {"%NAME": "definiteness", "%RANGE": "uima.cas.String"},
        "value": {"%NAME": "value", "%RANGE": "uima.cas.String"},
        "person": {"%NAME": "person", "%RANGE": "uima.cas.String"},
        "aspect": {"%NAME": "aspect", "%RANGE": "uima.cas.String"},
        "animacy": {"%NAME": "animacy", "%RANGE": "uima.cas.String"},
        "negative": {"%NAME": "negative", "%RANGE": "uima.cas.String"},
        "numType": {"%NAME": "numType", "%RANGE": "uima.cas.String"},
        "possessive": {"%NAME": "possessive", "%RANGE": "uima.cas.String"},
        "pronType": {"%NAME": "pronType", "%RANGE": "uima.cas.String"},
        "reflex": {"%NAME": "reflex", "%RANGE": "uima.cas.String"},
        "transitivity": {"%NAME": "transitivity", "%RANGE": "uima.cas.String"},
    },
    "de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS": {
        "%NAME": "de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS",
        "%SUPER_TYPE": "uima.tcas.Annotation",
        "PosValue": {"%NAME": "PosValue", "%RANGE": "uima.cas.String"},
        "coarseValue": {"%NAME": "coarseValue", "%RANGE": "uima.cas.String"},
    },
    "de.tudarmstadt.ukp.dkpro.core.api.metadata.type.DocumentMetaData": {
        "%NAME": "de.tudarmstadt.ukp.dkpro.core.api.metadata.type.DocumentMetaData",
        "%SUPER_TYPE": "uima.tcas.DocumentAnnotation",
        "documentTitle": {"%NAME": "documentTitle", "%RANGE": "uima.cas.String"},
        "documentId": {"%NAME": "documentId", "%RANGE": "uima.cas.String"},
        "documentUri": {"%NAME": "documentUri", "%RANGE": "uima.cas.String"},
        "collectionId": {"%NAME": "collectionId", "%RANGE": "uima.cas.String"},
        "documentBaseUri": {
            "%NAME": "documentBaseUri",
            "%RANGE": "uima.cas.String",
        },
        "isLastSegment": {"%NAME": "isLastSegment", "%RANGE": "uima.cas.Boolean"},
    },
    "de.tudarmstadt.ukp.dkpro.core.api.metadata.type.TagDescription": {
        "%NAME": "de.tudarmstadt.ukp.dkpro.core.api.metadata.type.TagDescription",
        "%SUPER_TYPE": "uima.cas.TOP",
        "name": {"%NAME": "name", "%RANGE": "uima.cas.String"},
    },
    "de.tudarmstadt.ukp.dkpro.core.api.metadata.type.TagsetDescription": {
        "%NAME": "de.tudarmstadt.ukp.dkpro.core.api.metadata.type.TagsetDescription",
        "%SUPER_TYPE": "uima.tcas.Annotation",
        "layer": {"%NAME": "layer", "%RANGE": "uima.cas.String"},
        "name": {"%NAME": "name", "%RANGE": "uima.cas.String"},
        "tags": {
            "%NAME": "tags",
            "%RANGE": "de.tudarmstadt.ukp.dkpro.core.api.metadata.type.TagDescription[]",
        },
        "componentName": {"%NAME": "componentName", "%RANGE": "uima.cas.String"},
        "modelLocation": {"%NAME": "modelLocation", "%RANGE": "uima.cas.String"},
        "modelVariant": {"%NAME": "modelVariant", "%RANGE": "uima.cas.String"},
        "modelLanguage": {"%NAME": "modelLanguage", "%RANGE": "uima.cas.String"},
        "modelVersion": {"%NAME": "modelVersion", "%RANGE": "uima.cas.String"},
        "input": {"%NAME": "input", "%RANGE": "uima.cas.Boolean"},
    },
    "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Lemma": {
        "%NAME": "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Lemma",
        "%SUPER_TYPE": "uima.tcas.Annotation",
        "value": {"%NAME": "value", "%RANGE": "uima.cas.String"},
    },
    "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence": {
        "%NAME": "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence",
        "%SUPER_TYPE": "uima.tcas.Annotation",
        "id": {"%NAME": "id", "%RANGE": "uima.cas.String"},
    },
    "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Stem": {
        "%NAME": "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Stem",
        "%SUPER_TYPE": "uima.tcas.Annotation",
        "value": {"%NAME": "value", "%RANGE": "uima.cas.String"},
    },
    "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token": {
        "%NAME": "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token",
        "%SUPER_TYPE": "uima.tcas.Annotation",
        "parent": {"%NAME": "parent", "%RANGE": "uima.tcas.Annotation"},
        "lemma": {
            "%NAME": "lemma",
            "%RANGE": "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Lemma",
        },
        "stem": {
            "%NAME": "stem",
            "%RANGE": "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Stem",
        },
        "pos": {
            "%NAME": "pos",
            "%RANGE": "de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS",
        },
        "morph": {
            "%NAME": "morph",
            "%RANGE": "de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.morph.MorphologicalFeatures",
        },
        "id": {"%NAME": "id", "%RANGE": "uima.cas.String"},
        "form": {
            "%NAME": "form",
            "%RANGE": "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.TokenForm",
        },
        "syntacticFunction": {
            "%NAME": "syntacticFunction",
            "%RANGE": "uima.cas.String",
        },
        "order": {"%NAME": "order", "%RANGE": "uima.cas.Integer"},
    },
    "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.TokenForm": {
        "%NAME": "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.TokenForm",
        "%SUPER_TYPE": "uima.tcas.Annotation",
        "value": {"%NAME": "value", "%RANGE": "uima.cas.String"},
    },
}


def converter_to_uima_json(
    doc: Doc,
    entity_mapping: Dict[str, Dict[str, str]],
    project_name: str,
    project_slug: str,
    entity_type_definition: Dict[str, Dict[str, Any]],
):
    entity_type_name, entity_def = next(iter(entity_type_definition.items()))
    entity_layer_ui_name = entity_def.get("uiName", entity_type_name.split(".")[-1])
    entity_features = [k for k in entity_def.keys() if not k.startswith("%")]

    layer_id = len(entity_features) + 1
    features = [
        {
            "%ID": layer_id,
            "%TYPE": "de.tudarmstadt.ukp.clarin.webanno.api.type.LayerDefinition",
            "name": entity_type_name,
            "uiName": entity_layer_ui_name,
        },
    ]

    feature_id = 1
    for feature_name in entity_features:
        features.append(
            {
                "%ID": feature_id,
                "%TYPE": "de.tudarmstadt.ukp.clarin.webanno.api.type.FeatureDefinition",
                "@layer": layer_id,
                "name": feature_name,
                "uiName": feature_name,
            }
        )
        feature_id += 1

    sofa_id = layer_id + 1
    doc_meta_id = sofa_id + 1
    tagset_id = doc_meta_id + 1

    features.append(
        {
            "%ID": sofa_id,
            "%TYPE": "uima.cas.Sofa",
            "sofaNum": 1,
            "sofaID": "_InitialView",
            "mimeType": "text",
            "sofaString": doc.text,
        }
    )
    features.append(
        {
            "%ID": doc_meta_id,
            "%TYPE": "de.tudarmstadt.ukp.dkpro.core.api.metadata.type.DocumentMetaData",
            "@sofa": sofa_id,
            "begin": 0,
            "end": len(doc.text),
            "language": "x-unspecified",
            "documentTitle": doc._.note_id,
            "documentId": doc._.note_id,
            "documentUri": f"{project_slug}/{doc._.note_id}",
            "collectionId": project_name,
            "documentBaseUri": project_name,
            "isLastSegment": False,
        },
    )
    i = tagset_id + 1
    for ent in doc.ents:
        if ent.label_ in entity_mapping:
            features.append(
                {
                    "%ID": i,
                    "%TYPE": entity_type_name,
                    "@sofa": sofa_id,
                    "begin": ent.start_char,
                    "end": ent.end_char,
                    **entity_mapping[ent.label_],
                }
            )
        i += 1
    return {
        FILENAME: f"{doc._.note_id}",
        "%TYPES": {**BASE_TYPE_DEFINITIONS, **entity_type_definition},
        "%FEATURE_STRUCTURES": features,
        "%VIEWS": {
            "_InitialView": {
                "%SOFA": sofa_id,
                "%MEMBERS": [f["%ID"] for f in features],
            }
        },
    }


def context_getter(doc: Doc, min_words: int = 80):
    start = None
    word_count = 0

    for sent in doc.sents:
        # Mark the start of the chunk
        if start is None:
            start = sent.start

        # Count non-punct tokens in this sentence
        word_count += sum(1 for token in sent if not token.is_punct)

        # If threshold reached, yield span
        if word_count >= min_words:
            yield doc[start : sent.end]
            start = None
            word_count = 0

    # Yield final remainder, if any
    if start is not None:
        yield doc[start:]




# EDS-NLP util to create documents from Markdown or XML markup.
# This has nothing to do with the LLM component itself.
conv = edsnlp.data.converters.MarkupToDocConverter(preset="xml")

def prepare_few_shot_samples(texts: List[str]) -> List[Doc]:
    return [conv(text) for text in texts]


def make_context_getter(min_words: int) -> Callable[[Doc], Callable]:
    return lambda doc: context_getter(doc, min_words=min_words)


def build_nlp_pipeline(
    prompt_text: str,
    api_url: str,
    model: str,
    few_shot_examples: List[Doc],
    max_few_shot_examples: int,
    max_concurrent_requests: int,
    temperature: float,
    min_words_per_chunk: int,
):
    nlp = edsnlp.blank("eds")
    nlp.add_pipe(eds.sentences())
    nlp.add_pipe(
        eds.llm_markup_extractor(
            api_url=api_url,
            model=model,
            examples=few_shot_examples,
            context_getter=make_context_getter(min_words=min_words_per_chunk),
            prompt=prompt_text,
            markup_mode="md",
            use_retriever=False,
            max_few_shot_examples=max_few_shot_examples,
            max_concurrent_requests=max_concurrent_requests,
            api_kwargs={"temperature": temperature},
        )
    )
    return nlp


app = Cli(pretty_exceptions_show_locals=False)


@app.command(name="preannotate")
def preannotate(
    *,
    input_path: Path,
    output_path: Path,
    uima_output_path: Path,
    api_url: str = "https://api.mistral.ai/v1/",
    model: str = "mistral-medium-2508",
    prompt_text: str,
    entity_mapping: Dict[str, Dict[str, str]],
    entity_type_definition: Dict[str, Dict[str, Any]],
    few_shot_examples: Optional[List[str]] = None,
    drop_ratio: float = 0.05,
    seed: Optional[int] = 42,
    project_name: str,
    project_slug: str,
    max_few_shot_examples: int = 3,
    max_concurrent_requests: int = 10,
    temperature: float = 0.0,
    min_words_per_chunk: int = 80,
    remove_pseudo_antipatterns: bool = False,
):
    """
    Pre-annotate NER dataset.

    Parameters
    ----------
    input_path: Path
        The path to the input standoff dataset.
    output_path: Path
        The path to write the pre-annotated standoff dataset.
    uima_output_path: Path
        The path to write the UIMA CAS JSON dataset export.
    api_url: str
        The LLM API URL.
    model: str
        The LLM model name.
    prompt_text: str
        The prompt text to use.
    entity_mapping: Dict[str, Dict[str, str]]
        The entity mapping for UIMA conversion.
    entity_type_definition: Dict[str, Dict[str, Any]]
        The entity type definition for the UIMA export.
    few_shot_examples: Optional[List[str]]
        Custom few-shot examples (as XML/markdown strings) for the LLM extractor.
    drop_ratio: float
        The ratio of entities to drop randomly.
    seed: Optional[int]
        The random seed for entity dropping.
    project_name: str
        The UIMA project name.
    project_slug: str
        The UIMA project slug.
    max_few_shot_examples: int
        The maximum number of few-shot examples to use.
    max_concurrent_requests: int
        The maximum number of concurrent requests to the LLM API.
    temperature: float
        The temperature for the LLM API.
    min_words_per_chunk: int
        The minimum number of words per chunk for context.
    remove_pseudo_antipatterns: bool
        Whether to remove pseudo-antipattern entities (e.g., dates like "J2").
    """
    resolved_few_shot = few_shot_examples
    few_shot_docs = prepare_few_shot_samples(resolved_few_shot)

    nlp = build_nlp_pipeline(
        prompt_text=prompt_text,
        api_url=api_url,
        model=model,
        few_shot_examples=few_shot_docs,
        max_few_shot_examples=max_few_shot_examples,
        max_concurrent_requests=max_concurrent_requests,
        temperature=temperature,
        min_words_per_chunk=min_words_per_chunk,
    )

    data = edsnlp.data.read_standoff(str(input_path), keep_txt_only_docs=True)
    data = data.map_pipeline(nlp)
    data = data.set_processing(show_progress=True)
    print("Writing complete pre-annotated data to", output_path)
    edsnlp.data.write_json(
        data,
        path=str(output_path),
        converter="standoff",
        overwrite=True,
    )

    annotated_data = edsnlp.data.read_json(str(output_path), converter="standoff")
    annotated_data = annotated_data.map(
        clean_and_drop_ents(seed=seed, drop_ratio=drop_ratio, remove_pseudo_antipatterns=remove_pseudo_antipatterns)
    )
    converter = lambda doc: converter_to_uima_json(  # noqa: E731
        doc=doc,
        entity_mapping=entity_mapping,
        project_name=project_name,
        project_slug=project_slug,
        entity_type_definition=entity_type_definition,
    )
    print("Writing UIMA CAS JSON to", uima_output_path)
    edsnlp.data.write_json(
        annotated_data,
        path=str(uima_output_path),
        converter=converter,
        overwrite=True,
    )


if __name__ == "__main__":
    app()
