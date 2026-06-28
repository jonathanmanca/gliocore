"""
agents/research_agent.py — agente AI per ricerca letteratura e valutazione modelli.

Usa Claude API per:
1. Cercare metodi di segmentazione rilevanti per il caso clinico
2. Interpretare le metriche e confrontare i risultati tra modelli
3. Generare un testo clinico di sintesi del run
"""
from __future__ import annotations
import json
import logging
from typing import Callable

import httpx

from config.settings import (
    ANTHROPIC_MODEL, AGENT_MAX_TOKENS,
    ANTHROPIC_API_KEY, ANTHROPIC_API_VERSION,
)

log = logging.getLogger(__name__)

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"


def call_anthropic(
    system: str,
    prompt: str,
    on_token: Callable[[str], None] | None = None,
) -> str:
    """
    Chiamata unica all'API Anthropic, condivisa dagli agenti.

    Legge la chiave da config (env ANTHROPIC_API_KEY) e invia gli header
    obbligatori (x-api-key, anthropic-version). Restituisce il testo della
    risposta, oppure un messaggio '[Agent error: ...]' leggibile in caso di
    errore (nessuna eccezione propagata all'UI).
    """
    if not ANTHROPIC_API_KEY:
        return ("[Agent error: ANTHROPIC_API_KEY non impostata. "
                "Imposta la variabile d'ambiente con la tua chiave Anthropic "
                "e riavvia GlioCore.]")

    payload = {
        "model":      ANTHROPIC_MODEL,
        "max_tokens": AGENT_MAX_TOKENS,
        "system":     system,
        "messages":   [{"role": "user", "content": prompt}],
    }
    headers = {
        "x-api-key":         ANTHROPIC_API_KEY,
        "anthropic-version": ANTHROPIC_API_VERSION,
        "content-type":      "application/json",
    }
    try:
        with httpx.Client(timeout=60) as client:
            response = client.post(ANTHROPIC_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data["content"][0]["text"]
    except httpx.HTTPStatusError as e:
        # Messaggio API utile (es. 401 chiave errata, 429 rate limit)
        detail = e.response.text[:300] if e.response is not None else str(e)
        log.error(f"Anthropic API HTTP {e.response.status_code}: {detail}")
        return f"[Agent error: HTTP {e.response.status_code} — {detail}]"
    except Exception as e:
        log.error(f"Anthropic API error: {e}")
        return f"[Agent error: {e}]"


class ResearchAgent:
    """
    Agente di ricerca: dato il tipo di dato e il paziente,
    suggerisce metodi di segmentazione dalla letteratura.
    """

    SYSTEM_PROMPT = """You are an agent specialized in oncological neuroimaging and segmentation 
of gliomas. You have access to up-to-date knowledge of PET/MRI segmentation methods.
Always respond in English, concisely and oriented to clinical practice.
When you suggest methods, always specify: name, essential bibliographic reference, 
main advantage for this type of data, and main limitation."""

    def suggest_methods(
        self,
        modality: str,
        n_patients: int,
        current_models: list[str],
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        """
        Suggerisce metodi di segmentazione adatti al dataset.

        Parameters
        ----------
        modality       : es. "PET-T1 coregistrata (SUV/SUVR)"
        n_patients     : numero di pazienti nel dataset
        current_models : modelli già implementati
        on_token       : callback per streaming token nell'UI
        """
        prompt = (
            f"I have a dataset of {n_patients} glioma patients, images: {modality}. "
            f"I have already implemented: {', '.join(current_models)}. "
            "Suggest 3 additional segmentation methods suitable for this type of data, "
            "ordered by ease of implementation in Python. "
            "For each: name, reference paper, pros, cons, Python library."
        )
        return self._call_api(self.SYSTEM_PROMPT, prompt, on_token)

    def search_literature(
        self,
        query: str,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        """Ricerca letteratura su un argomento specifico."""
        prompt = (
            f"Search your knowledge for recent literature (2020-2025) on: {query}. "
            "Return: 3-5 relevant papers with authors, year, title, DOI if known, "
            "and one summary sentence for each."
        )
        return self._call_api(self.SYSTEM_PROMPT, prompt, on_token)

    def _call_api(
        self,
        system: str,
        prompt: str,
        on_token: Callable[[str], None] | None,
    ) -> str:
        return call_anthropic(system, prompt, on_token)


class ModelAgent:
    """
    Agente di valutazione modelli: interpreta metriche e confronta risultati.
    """

    SYSTEM_PROMPT = """You are an expert in medical image segmentation and model 
validation. You interpret quantitative segmentation metrics (BIC, AIC, Dice, 
XB index, FPC) in a clinically relevant way. Respond in English, in a 
concise way, highlighting the practical implications for the neuroradiologist."""

    def interpret_metrics(
        self,
        patient_id: str,
        model_name: str,
        metrics: dict,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        """Interpreta le metriche di un run in linguaggio clinico."""
        prompt = (
            f"Patient: {patient_id}, Model: {model_name}\n"
            f"Metrics: {json.dumps(metrics, indent=2)}\n\n"
            "Interpret these results: did the model find a sensible number of clusters? "
            "Is the convergence reliable? What do the metrics suggest about the quality "
            "of the segmentation? Suggestions to improve?"
        )
        return self._call_api(self.SYSTEM_PROMPT, prompt, on_token)

    def compare_models(
        self,
        patient_id: str,
        results: list[dict],
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        """
        Confronta i risultati di più modelli sullo stesso paziente.

        results: lista di {model_name, n_clusters, metrics}
        """
        summary = json.dumps(results, indent=2)
        prompt = (
            f"Patient: {patient_id}\n"
            f"Model comparison:\n{summary}\n\n"
            "Which model produced the most reliable segmentation for this patient? "
            "Justify the answer with the metrics. "
            "Is there agreement among the models on the number of clusters? "
            "Which areas might require manual review?"
        )
        return self._call_api(self.SYSTEM_PROMPT, prompt, on_token)

    def generate_clinical_report(
        self,
        patient_id: str,
        model_name: str,
        n_clusters: int,
        metrics: dict,
        wm_tracts: list[dict] | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        """
        Genera una sintesi clinica del risultato di segmentazione.
        Include coinvolgimento fasci WM se disponibile.
        """
        wm_section = ""
        if wm_tracts:
            wm_section = f"\nInvolved WM tracts:\n{json.dumps(wm_tracts, indent=2)}"

        prompt = (
            f"Patient: {patient_id}, Model: {model_name}\n"
            f"Clusters found: {n_clusters} "
            f"(1=hypometabolism/necrosis, {n_clusters}=hypermetabolism/infiltration)\n"
            f"Metrics: {json.dumps(metrics)}{wm_section}\n\n"
            "Write a structured clinical summary (max 200 words) that a neuroradiologist "
            "could include in the report. Include: method used, number of clusters, "
            "metabolic interpretation, any WM tracts affected, limitations."
        )
        return self._call_api(self.SYSTEM_PROMPT, prompt, on_token)

    def _call_api(
        self,
        system: str,
        prompt: str,
        on_token: Callable[[str], None] | None,
    ) -> str:
        return call_anthropic(system, prompt, on_token)
