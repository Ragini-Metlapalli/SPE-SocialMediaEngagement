import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Send, MapPin, Hash, Layout, Type } from 'lucide-react';
import axios from 'axios';

const EngagementForm = ({ onResult }) => {
    const [loading, setLoading] = useState(false);
    const [loadingLocation, setLoadingLocation] = useState(true);
    const [formData, setFormData] = useState({
        caption: '',
        platform: 'Instagram',
        hashtags: '',
        location: 'Detecting...'
    });

    React.useEffect(() => {
        if ("geolocation" in navigator) {
            navigator.geolocation.getCurrentPosition(async (position) => {
                const { latitude, longitude } = position.coords;
                try {
                    // Using BigDataCloud's free reverse geocoding API (client-side allowed)
                    const res = await axios.get(`https://api.bigdatacloud.net/data/reverse-geocode-client?latitude=${latitude}&longitude=${longitude}&localityLanguage=en`);
                    const city = res.data.city || res.data.locality || "Unknown Location";
                    setFormData(prev => ({ ...prev, location: city }));
                } catch (err) {
                    console.error("Geo API Error", err);
                    setFormData(prev => ({ ...prev, location: "Location Unavailable" }));
                } finally {
                    setLoadingLocation(false);
                }
            }, (error) => {
                console.error("Geolocation Error", error);
                setFormData(prev => ({ ...prev, location: "Permission Denied" }));
                setLoadingLocation(false);
            });
        } else {
            setFormData(prev => ({ ...prev, location: "Not Support" }));
            setLoadingLocation(false);
        }
    }, []);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        try {
            // TODO: Replace with environment variable
            const response = await axios.post('http://localhost:8000/predict', formData);
            onResult(response.data);
        } catch (error) {
            console.error("Error predicting engagement:", error);
            alert("Failed to get prediction. Ensure Backend is running on port 8000.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass-panel p-8 w-full max-w-lg mx-auto"
        >
            <form onSubmit={handleSubmit} className="space-y-6">
                <div>
                    <label className="block text-sm font-medium text-slate-300 mb-2 flex items-center gap-2">
                        <Type className="w-4 h-4 text-indigo-400" /> Caption
                    </label>
                    <textarea
                        required
                        rows="4"
                        className="input-field resize-none"
                        placeholder="Write your engaging possible caption here..."
                        value={formData.caption}
                        onChange={(e) => setFormData({ ...formData, caption: e.target.value })}
                    />
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                        <label className="block text-sm font-medium text-slate-300 mb-2 flex items-center gap-2">
                            <Layout className="w-4 h-4 text-pink-400" /> Platform
                        </label>
                        <select
                            className="input-field appearance-none"
                            value={formData.platform}
                            onChange={(e) => setFormData({ ...formData, platform: e.target.value })}
                        >
                            <option>Instagram</option>
                            <option>Twitter</option>
                            <option>Facebook</option>
                            <option>LinkedIn</option>
                        </select>
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-slate-300 mb-2 flex items-center gap-2">
                            <MapPin className="w-4 h-4 text-red-400" /> Location (Auto-Detected)
                        </label>
                        <div className="relative">
                            <input
                                type="text"
                                className="input-field pr-10"
                                placeholder="Detecting location..."
                                value={formData.location}
                                readOnly
                            />
                            {loadingLocation && (
                                <div className="absolute right-3 top-3 w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                            )}
                        </div>
                    </div>
                </div>

                <div>
                    <label className="block text-sm font-medium text-slate-300 mb-2 flex items-center gap-2">
                        <Hash className="w-4 h-4 text-blue-400" /> Hashtags
                    </label>
                    <input
                        type="text"
                        className="input-field"
                        placeholder="#marketing #social..."
                        value={formData.hashtags}
                        onChange={(e) => setFormData({ ...formData, hashtags: e.target.value })}
                    />
                </div>

                <button
                    type="submit"
                    disabled={loading}
                    className="btn-primary flex items-center justify-center gap-2 disabled:opacity-70 disabled:cursor-not-allowed"
                >
                    {loading ? (
                        <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                    ) : (
                        <>
                            Analyze & Optimize <Send className="w-4 h-4" />
                        </>
                    )}
                </button>
            </form>
        </motion.div>
    );
};

export default EngagementForm;
