import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Zap, TrendingUp, BarChart2 } from 'lucide-react';

const HomePage = () => {
    const navigate = useNavigate();

    return (
        <div className="min-h-screen bg-slate-900 text-white flex flex-col items-center justify-center p-6 relative overflow-hidden">

            {/* Background Gradients */}
            <div className="absolute top-0 left-0 w-96 h-96 bg-purple-600 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob"></div>
            <div className="absolute top-0 right-0 w-96 h-96 bg-blue-600 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob animation-delay-2000"></div>

            <div className="text-center z-10 max-w-4xl">
                <h1 className="text-6xl font-bold mb-6 bg-gradient-to-r from-blue-400 to-purple-500 text-transparent bg-clip-text">
                    Social Media Optimizer
                </h1>
                <p className="text-xl text-gray-300 mb-10 leading-relaxed">
                    Unlock the power of <span className="text-blue-400 font-semibold">AI & Deep NLP</span> to maximize your post engagement.
                    <br />
                    We analyze your content's sentiment, toxicity, and topic to predict the <span className="text-purple-400 font-semibold">Perfect Time to Post</span>.
                </p>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
                    <div className="p-6 bg-white/5 rounded-xl border border-white/10 backdrop-blur-sm hover:bg-white/10 transition">
                        <Zap className="w-10 h-10 text-yellow-400 mx-auto mb-4" />
                        <h3 className="text-lg font-semibold mb-2">Instant NLP Analysis</h3>
                        <p className="text-sm text-gray-400">Automatic sentiment, topic, and toxicity detection using Transformers.</p>
                    </div>
                    <div className="p-6 bg-white/5 rounded-xl border border-white/10 backdrop-blur-sm hover:bg-white/10 transition">
                        <TrendingUp className="w-10 h-10 text-green-400 mx-auto mb-4" />
                        <h3 className="text-lg font-semibold mb-2">Max Engagement</h3>
                        <p className="text-sm text-gray-400">Predict the exact day and hour to get maximum likes and shares.</p>
                    </div>
                    <div className="p-6 bg-white/5 rounded-xl border border-white/10 backdrop-blur-sm hover:bg-white/10 transition">
                        <BarChart2 className="w-10 h-10 text-blue-400 mx-auto mb-4" />
                        <h3 className="text-lg font-semibold mb-2">Smart Predictions</h3>
                        <p className="text-sm text-gray-400">Powered by LightGBM and trained on 150k+ real social media posts.</p>
                    </div>
                </div>

                <button
                    onClick={() => navigate('/analyze')}
                    className="group relative inline-flex items-center justify-center px-8 py-4 text-lg font-bold text-white transition-all duration-200 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-600 hover:from-blue-500 hover:to-purple-500 hover:scale-105 shadow-lg hover:shadow-purple-500/50"
                >
                    Start Analyzing
                    <svg className="w-5 h-5 ml-2 -mr-1 transition-transform group-hover:translate-x-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 7l5 5m0 0l-5 5m5-5H6" />
                    </svg>
                </button>
            </div>

            {/* Footer */}
            <div className="absolute bottom-6 text-gray-500 text-sm">
                Social Media Engagement Project | Sem 7
            </div>
        </div>
    );
};

export default HomePage;
