import React from 'react';
import { motion } from 'framer-motion';
import { Calendar, Clock, TrendingUp, AlertTriangle, Smile, Hash, MessageCircle } from 'lucide-react';

const MetricCard = ({ icon: Icon, label, value, color = "text-indigo-400" }) => (
    <div className="bg-slate-800/40 p-3 rounded-lg border border-slate-700/50 flex flex-col items-center text-center">
        <Icon className={`w-6 h-6 ${color} mb-2`} />
        <span className="text-xs text-slate-400 uppercase tracking-wider">{label}</span>
        <span className="font-semibold text-slate-100">{value}</span>
    </div>
);

const PredictionResult = ({ result }) => {
    if (!result) return null;

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass-panel p-6 mt-8 w-full max-w-2xl mx-auto"
        >
            <div className="text-center mb-8">
                <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-emerald-400 to-cyan-400">
                    Optimal Posting Time Found
                </h2>
                <p className="text-slate-400 text-sm mt-1">Based on simulated audience activity</p>
            </div>

            <div className="flex justify-center gap-6 mb-8">
                <div className="text-center">
                    <div className="text-4xl font-bold text-white mb-1">{result.recommended_day}</div>
                    <div className="flex items-center justify-center text-indigo-300 gap-1 text-sm">
                        <Calendar className="w-4 h-4" /> Recommended Day
                    </div>
                </div>
                <div className="w-px bg-slate-700/50"></div>
                <div className="text-center">
                    <div className="text-4xl font-bold text-white mb-1">{result.recommended_time}</div>
                    <div className="flex items-center justify-center text-indigo-300 gap-1 text-sm">
                        <Clock className="w-4 h-4" /> Time Slot
                    </div>
                </div>
            </div>

            <div className="space-y-4">
                <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-2">Content Analysis</h3>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    <MetricCard
                        icon={TrendingUp}
                        label="Sentiment"
                        value={`${result.sentiment_label} (${(result.sentiment_score * 100).toFixed(0)}%)`}
                        color={result.sentiment_label === 'Positive' ? 'text-emerald-400' : 'text-amber-400'}
                    />
                    <MetricCard
                        icon={Smile}
                        label="Emotion"
                        value={result.emotion_type}
                        color="text-pink-400"
                    />
                    <MetricCard
                        icon={AlertTriangle}
                        label="Toxicity"
                        value={(result.toxicity_score * 100).toFixed(1) + '%'}
                        color={result.toxicity_score > 0.1 ? 'text-red-400' : 'text-emerald-400'}
                    />
                    <MetricCard
                        icon={MessageCircle}
                        label="Topics"
                        value={result.topic_categories[0]}
                        color="text-blue-400"
                    />
                </div>

                <div className="bg-slate-800/40 p-4 rounded-lg border border-slate-700/50 mt-4">
                    <div className="flex items-start gap-2">
                        <Hash className="w-5 h-5 text-slate-400 mt-1" />
                        <div>
                            <span className="text-xs text-slate-400 uppercase tracking-wider block mb-1">Generated Keywords</span>
                            <div className="flex flex-wrap gap-2">
                                {result.keywords.map((kw, i) => (
                                    <span key={i} className="px-2 py-1 bg-indigo-500/20 text-indigo-300 text-xs rounded-full border border-indigo-500/30">
                                        #{kw}
                                    </span>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </motion.div>
    );
};

export default PredictionResult;
