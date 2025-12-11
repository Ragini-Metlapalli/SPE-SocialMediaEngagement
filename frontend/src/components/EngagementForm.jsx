import React, { useState } from 'react';
import axios from 'axios';
import { ArrowLeft, Loader2, CheckCircle, AlertTriangle } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const EngagementForm = () => {
    const navigate = useNavigate();
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);

    const [formData, setFormData] = useState({
        platform: 'Twitter',
        caption: '',
        followers: 1000,
        account_age_days: 365,
        location: 'North America',
        media_type: 'Text',
        verified: false,
        cross_platform_spread: false
    });

    const locations = ["North America", "Europe", "Asia", "South America", "Africa", "Oceania", "Unknown"];
    const platforms = ["Twitter", "Instagram", "Facebook", "Reddit", "TikTok", "YouTube"];
    const mediaTypes = ["Text", "Image", "Video", "Link", "Poll"];

    const handleChange = (e) => {
        const { name, value, type, checked } = e.target;
        setFormData(prev => ({
            ...prev,
            [name]: type === 'checkbox' ? checked : value
        }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError(null);
        setResult(null);

        try {
            // prepare payload
            const payload = {
                ...formData,
                verified: formData.verified ? 1 : 0,
                cross_platform_spread: formData.cross_platform_spread ? 1 : 0,
                followers: parseInt(formData.followers),
                account_age_days: parseInt(formData.account_age_days)
            };

            const response = await axios.post('/api/predict', payload);
            setResult(response.data);
        } catch (err) {
            console.error(err);
            setError('Failed to analyze. Please ensure Backend is running.');
        } finally {
            setLoading(false);
        }
    };

    const days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];

    return (
        <div className="min-h-screen bg-slate-900 text-gray-100 py-12 px-4 sm:px-6 lg:px-8 font-sans">

            <button
                onClick={() => navigate('/')}
                className="absolute top-6 left-6 flex items-center text-gray-400 hover:text-white transition"
            >
                <ArrowLeft className="w-5 h-5 mr-2" /> Back to Home
            </button>

            <div className="max-w-5xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-12">

                {/* FORM SECTION */}
                <div>
                    <div className="mb-8">
                        <h2 className="text-3xl font-bold text-white mb-2">Analyze Content</h2>
                        <p className="text-gray-400">Fill in the details to get AI-powered predictions.</p>
                    </div>

                    <form onSubmit={handleSubmit} className="space-y-6 bg-white/5 p-8 rounded-2xl border border-white/10 backdrop-blur-sm">

                        {/* Platform & Media */}
                        <div className="grid grid-cols-2 gap-6">
                            <div>
                                <label className="block text-sm font-medium text-gray-300 mb-2">Platform</label>
                                <select name="platform" value={formData.platform} onChange={handleChange}
                                    className="w-full bg-slate-800 border-none rounded-lg p-3 text-white focus:ring-2 focus:ring-blue-500">
                                    {platforms.map(p => <option key={p} value={p}>{p}</option>)}
                                </select>
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-gray-300 mb-2">Media Type</label>
                                <select name="media_type" value={formData.media_type} onChange={handleChange}
                                    className="w-full bg-slate-800 border-none rounded-lg p-3 text-white focus:ring-2 focus:ring-blue-500">
                                    {mediaTypes.map(m => <option key={m} value={m}>{m}</option>)}
                                </select>
                            </div>
                        </div>

                        {/* Caption */}
                        <div>
                            <label className="block text-sm font-medium text-gray-300 mb-2">Post Caption</label>
                            <textarea
                                name="caption"
                                rows="4"
                                value={formData.caption}
                                onChange={handleChange}
                                placeholder="Enter your post content here (hashtags supported)..."
                                className="w-full bg-slate-800 border-none rounded-lg p-3 text-white focus:ring-2 focus:ring-blue-500"
                                required
                            />
                        </div>

                        {/* Stats Row 1 */}
                        <div className="grid grid-cols-2 gap-6">
                            <div>
                                <label className="block text-sm font-medium text-gray-300 mb-2">Follower Count</label>
                                <input type="number" name="followers" value={formData.followers} onChange={handleChange}
                                    className="w-full bg-slate-800 border-none rounded-lg p-3 text-white focus:ring-2 focus:ring-blue-500" />
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-gray-300 mb-2">Account Age (Days)</label>
                                <input type="number" name="account_age_days" value={formData.account_age_days} onChange={handleChange}
                                    className="w-full bg-slate-800 border-none rounded-lg p-3 text-white focus:ring-2 focus:ring-blue-500" />
                            </div>
                        </div>

                        {/* Country */}
                        <div>
                            <label className="block text-sm font-medium text-gray-300 mb-2">Target Audience Location</label>
                            <select name="location" value={formData.location} onChange={handleChange}
                                className="w-full bg-slate-800 border-none rounded-lg p-3 text-white focus:ring-2 focus:ring-blue-500">
                                {locations.map(l => <option key={l} value={l}>{l}</option>)}
                            </select>
                        </div>

                        {/* Toggles */}
                        <div className="flex gap-8">
                            <label className="flex items-center space-x-3 cursor-pointer">
                                <input type="checkbox" name="verified" checked={formData.verified} onChange={handleChange}
                                    className="w-5 h-5 rounded border-gray-600 text-blue-500 focus:ring-blue-500 bg-slate-800" />
                                <span className="text-gray-300">Verified Account</span>
                            </label>
                            <label className="flex items-center space-x-3 cursor-pointer">
                                <input type="checkbox" name="cross_platform_spread" checked={formData.cross_platform_spread} onChange={handleChange}
                                    className="w-5 h-5 rounded border-gray-600 text-purple-500 focus:ring-purple-500 bg-slate-800" />
                                <span className="text-gray-300">Cross Platform?</span>
                            </label>
                        </div>

                        <button type="submit" disabled={loading}
                            className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white font-bold py-4 rounded-xl shadow-lg hover:shadow-blue-500/30 transition transform hover:scale-[1.02] disabled:opacity-50 disabled:cursor-not-allowed flex justify-center items-center">
                            {loading ? <><Loader2 className="animate-spin mr-2" /> Analyzing...</> : 'Analyze & Optimize'}
                        </button>
                    </form>
                </div>

                {/* RESULTS SECTION */}
                <div className="flex flex-col justify-center">
                    {error && (
                        <div className="p-4 bg-red-500/20 border border-red-500/50 rounded-xl text-red-200 flex items-center mb-6">
                            <AlertTriangle className="mr-3" /> {error}
                        </div>
                    )}

                    {!result && !loading && (
                        <div className="text-center text-gray-500 p-12 border-2 border-dashed border-gray-700 rounded-2xl">
                            <h3 className="text-xl font-semibold mb-2">Ready to Predict</h3>
                            <p>Enter your post details to see AI insights here.</p>
                        </div>
                    )}

                    {result && (
                        <div className="space-y-6 animate-fade-in-up">

                            {/* Best Time Card */}
                            <div className="bg-gradient-to-br from-indigo-900 to-slate-900 rounded-2xl p-8 border border-indigo-500/30 shadow-2xl relative overflow-hidden">
                                <div className="absolute top-0 right-0 p-32 bg-blue-500/10 rounded-full blur-3xl -mr-16 -mt-16"></div>

                                <h3 className="text-indigo-300 text-sm font-bold uppercase tracking-wider mb-1">Recommended Post Time</h3>
                                <div className="flex items-baseline space-x-4 mb-4">
                                    <span className="text-5xl font-extrabold text-white">{days[result.best_day]}</span>
                                    <span className="text-3xl text-indigo-400">@ {result.best_hour}:00</span>
                                </div>
                                <div className="inline-flex items-center px-3 py-1 bg-green-500/20 text-green-400 rounded-full text-sm font-medium">
                                    <CheckCircle className="w-4 h-4 mr-1" /> Low Competition Window
                                </div>
                            </div>

                            {/* NLP Insights Grid */}
                            <h3 className="text-xl font-semibold text-white mt-4 mb-2">AI Content Analysis</h3>
                            <div className="grid grid-cols-2 gap-4">
                                <div className="bg-white/5 p-4 rounded-xl border border-white/5">
                                    <div className="text-gray-400 text-sm mb-1">Detected Topic</div>
                                    <div className="text-xl font-bold text-blue-400">{result.nlp_insights.topic}</div>
                                </div>
                                <div className="bg-white/5 p-4 rounded-xl border border-white/5">
                                    <div className="text-gray-400 text-sm mb-1">Sentiment</div>
                                    <div className={`text-xl font-bold capitalize ${result.nlp_insights.sentiment_category === 'positive' ? 'text-green-400' :
                                        result.nlp_insights.sentiment_category === 'negative' ? 'text-red-400' : 'text-yellow-400'
                                        }`}>
                                        {result.nlp_insights.sentiment_category}
                                    </div>
                                </div>
                                <div className="bg-white/5 p-4 rounded-xl border border-white/5">
                                    <div className="text-gray-400 text-sm mb-1">Toxicity Score</div>
                                    <div className="text-xl font-bold text-orange-400">
                                        {result.nlp_insights.toxicity_score.toFixed(1)}%
                                    </div>
                                </div>
                                <div className="bg-white/5 p-4 rounded-xl border border-white/5">
                                    <div className="text-gray-400 text-sm mb-1">Language</div>
                                    <div className="text-xl font-bold text-gray-200 uppercase">{result.nlp_insights.language}</div>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default EngagementForm;
