import { useState } from "react";

import PredictorForm from "./components/PredictorForm";
import ResultDisplay from "./components/ResultDisplay";

export default function App() {
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  return (
    <main className="min-h-screen bg-slate-950 px-4 py-8 text-slate-100">
      <div className="mx-auto max-w-6xl">
        <header className="mb-8 border-b border-slate-800 pb-6">
          <p className="font-mono text-sm uppercase tracking-[0.4em] text-emerald-300">
            FORESIGHT
          </p>
          <h1 className="mt-3 text-3xl font-bold tracking-wide md:text-5xl">
            AI Probabilistic Forecasting Engine
          </h1>
        </header>

        <div className="grid gap-6 lg:grid-cols-[minmax(0,0.95fr)_minmax(0,1.05fr)]">
          <PredictorForm
            onResult={setResult}
            onLoadingChange={setIsLoading}
          />

          <div className="min-h-[420px]">
            {isLoading ? (
              <div className="flex h-full min-h-[420px] items-center justify-center rounded-lg border border-slate-800 bg-slate-950/80 font-mono text-emerald-300">
                Running probabilistic analysis...
              </div>
            ) : (
              <ResultDisplay result={result} />
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
