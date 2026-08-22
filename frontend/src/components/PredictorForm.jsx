import { useState } from "react";

export default function PredictorForm({ onResult, onLoadingChange }) {
  const [eventQuestion, setEventQuestion] = useState("");
  const [manualCrowdProb, setManualCrowdProb] = useState(50);
  const [contextArticles, setContextArticles] = useState("");
  const [error, setError] = useState("");

  async function handleSubmit(event) {
    event.preventDefault();
    setError("");
    onLoadingChange(true);

    try {
      const response = await fetch("http://localhost:8000/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          event_question: eventQuestion,
          manual_crowd_prob: Number(manualCrowdProb) / 100,
          context_articles: contextArticles
            .split(/\n\s*\n/)
            .map((article) => article.trim())
            .filter(Boolean),
        }),
      });

      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || "Prediction request failed.");
      }

      const data = await response.json();
      onResult(data);
    } catch (err) {
      setError(err.message);
      onResult(null);
    } finally {
      onLoadingChange(false);
    }
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="rounded-lg border border-emerald-500/30 bg-black/80 p-6 font-mono shadow-2xl shadow-emerald-950/40"
    >
      <div className="mb-5 flex items-center justify-between border-b border-emerald-500/20 pb-3">
        <span className="text-sm uppercase tracking-widest text-emerald-300">
          predictor@foresight:~$
        </span>
        <span className="text-xs text-slate-500">Phase 1 MVP</span>
      </div>

      <label className="mb-5 block">
        <span className="mb-2 block text-sm text-slate-300">
          Future Event Question
        </span>
        <input
          value={eventQuestion}
          onChange={(event) => setEventQuestion(event.target.value)}
          required
          type="text"
          placeholder="Will the event resolve YES by the target date?"
          className="w-full rounded border border-slate-700 bg-slate-950 px-3 py-3 text-slate-100 outline-none transition focus:border-emerald-400"
        />
      </label>

      <label className="mb-5 block">
        <span className="mb-2 block text-sm text-slate-300">
          Manual Crowd Probability
        </span>
        <div className="flex items-center gap-3">
          <input
            value={manualCrowdProb}
            onChange={(event) => setManualCrowdProb(event.target.value)}
            required
            type="number"
            min="0"
            max="100"
            step="0.1"
            className="w-32 rounded border border-slate-700 bg-slate-950 px-3 py-3 text-slate-100 outline-none transition focus:border-emerald-400"
          />
          <span className="text-slate-400">%</span>
        </div>
      </label>

      <label className="mb-6 block">
        <span className="mb-2 block text-sm text-slate-300">
          Paste Context Articles Here
        </span>
        <textarea
          value={contextArticles}
          onChange={(event) => setContextArticles(event.target.value)}
          rows="12"
          placeholder="Paste one or more articles, separated by blank lines..."
          className="w-full resize-y rounded border border-slate-700 bg-slate-950 px-3 py-3 text-slate-100 outline-none transition focus:border-emerald-400"
        />
      </label>

      {error ? (
        <div className="mb-4 rounded border border-red-500/40 bg-red-950/40 px-3 py-2 text-sm text-red-200">
          {error}
        </div>
      ) : null}

      <button
        type="submit"
        className="w-full rounded bg-emerald-400 px-4 py-3 font-bold uppercase tracking-widest text-slate-950 transition hover:bg-emerald-300 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
        disabled={!eventQuestion.trim()}
      >
        Run Analysis
      </button>
    </form>
  );
}
