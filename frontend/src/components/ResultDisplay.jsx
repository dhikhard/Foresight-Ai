function formatPercent(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

function ProbabilityBar({ label, value, colorClass }) {
  const percent = Math.max(0, Math.min(100, Number(value || 0) * 100));

  return (
    <div>
      <div className="mb-2 flex items-center justify-between font-mono text-sm text-slate-300">
        <span>{label}</span>
        <span>{formatPercent(value)}</span>
      </div>
      <div className="h-4 overflow-hidden rounded bg-slate-800">
        <div
          className={`h-full ${colorClass} transition-all duration-500`}
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
  );
}

export default function ResultDisplay({ result }) {
  if (!result) {
    return null;
  }

  const edge = Number(result.information_edge || 0);
  const edgeClass = edge >= 0 ? "text-cyan-300" : "text-red-300";

  return (
    <section className="rounded-lg border border-slate-700 bg-slate-950/90 p-6 shadow-2xl shadow-black/30">
      <div className="mb-6 flex flex-col gap-2 border-b border-slate-800 pb-4 md:flex-row md:items-end md:justify-between">
        <div>
          <p className="font-mono text-xs uppercase tracking-widest text-slate-500">
            Forecast Output
          </p>
          <h2 className="mt-1 text-xl font-semibold text-slate-100">
            {result.event_question}
          </h2>
        </div>
        <div className="font-mono text-sm">
          <span className="text-slate-500">Information Edge </span>
          <span className={`text-lg font-bold ${edgeClass}`}>
            {edge >= 0 ? "+" : ""}
            {formatPercent(edge)}
          </span>
        </div>
      </div>

      <div className="grid gap-5 md:grid-cols-2">
        <ProbabilityBar
          label="AI Probability"
          value={result.ai_probability}
          colorClass="bg-cyan-400"
        />
        <ProbabilityBar
          label="Crowd Probability"
          value={result.crowd_probability}
          colorClass="bg-emerald-400"
        />
      </div>

      <div className="mt-6 rounded border border-slate-800 bg-black/60 p-4 font-mono">
        <h3 className="mb-2 text-sm uppercase tracking-widest text-slate-400">
          Reasoning
        </h3>
        <p className="whitespace-pre-wrap text-sm leading-6 text-slate-200">
          {result.reasoning || "No reasoning returned."}
        </p>
      </div>

      <div className="mt-4 rounded border border-slate-800 bg-black/60 p-4 font-mono">
        <h3 className="mb-3 text-sm uppercase tracking-widest text-slate-400">
          Citations
        </h3>
        {result.citations?.length ? (
          <ul className="space-y-3">
            {result.citations.map((citation, index) => (
              <li
                key={`${citation.quote}-${index}`}
                className="border-l-2 border-cyan-400 pl-3 text-sm leading-6 text-slate-300"
              >
                "{citation.quote}"
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-sm text-slate-500">No citations returned.</p>
        )}
      </div>
    </section>
  );
}
