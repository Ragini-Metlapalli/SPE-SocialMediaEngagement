import React, { useState } from 'react';
import EngagementForm from './components/EngagementForm';
import PredictionResult from './components/PredictionResult';
import { Sparkles } from 'lucide-react';

function App() {
  const [result, setResult] = useState(null);

  return (
    <div className="min-h-screen py-12 px-4 sm:px-6 lg:px-8 flex flex-col items-center">
      <div className="text-center mb-10">
        <div className="flex items-center justify-center gap-3 mb-4">
          <div className="p-3 bg-indigo-500/20 rounded-2xl border border-indigo-500/30">
            <Sparkles className="w-8 h-8 text-indigo-400" />
          </div>
        </div>
        <h1 className="text-4xl md:text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-indigo-300 via-white to-purple-300 tracking-tight">
          Social Pulse
        </h1>
        <p className="mt-4 text-lg text-slate-400 max-w-lg mx-auto leading-relaxed">
          AI-powered insights for your social media strategy. <br />
          Optimize posting times and engagement.
        </p>
      </div>

      <EngagementForm onResult={setResult} />

      <PredictionResult result={result} />

      <footer className="mt-16 text-slate-600 text-sm">
        © 2025 Prediction System. Powered by DevOps & ML.
      </footer>
    </div>
  );
}

export default App;
