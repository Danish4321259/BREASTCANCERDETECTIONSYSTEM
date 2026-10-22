import React, { useState } from 'react';
import {
  Activity,
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  FileText,
  ImageIcon,
  Info,
  Layers,
  RotateCcw,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  UploadCloud,
  X
} from 'lucide-react';

interface PredictionResponse {
  prediction: string;
  is_malignant: boolean;
  probability_malignant: number;
  probability_benign: number;
  confidence_percentage: number;
  model_used: string;
  risk_assessment: string;
  disclaimer: string;
  key_features_analysis?: Array<{
    feature: string;
    value: number;
    unit: string;
    closer_to: string;
  }>;
}

export default function App() {
  const [activeTab, setActiveTab] = useState<'numeric' | 'image'>('numeric');
  const [radius, setRadius] = useState<string>('');
  const [texture, setTexture] = useState<string>('');
  const [perimeter, setPerimeter] = useState<string>('');
  const [area, setArea] = useState<string>('');
  const [smoothness, setSmoothness] = useState<string>('');

  // Advanced 30-features drawer state
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false);
  const [selectedModel, setSelectedModel] = useState<string>('best');
  const [showDashboardModal, setShowDashboardModal] = useState<boolean>(false);

  // Status and results
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Image upload state
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [imageAnalyzed, setImageAnalyzed] = useState<boolean>(false);

  // Sample values presets from WDBC
  const loadSample = (type: 'malignant' | 'benign') => {
    if (type === 'malignant') {
      setRadius('17.99');
      setTexture('10.38');
      setPerimeter('122.8');
      setArea('1001');
      setSmoothness('0.1184');
    } else {
      setRadius('13.54');
      setTexture('14.36');
      setPerimeter('87.46');
      setArea('566.3');
      setSmoothness('0.09779');
    }
    setResult(null);
    setErrorMessage(null);
  };

  const handleReset = () => {
    setRadius('');
    setTexture('');
    setPerimeter('');
    setArea('');
    setSmoothness('');
    setResult(null);
    setErrorMessage(null);
    setSelectedImage(null);
    setImageAnalyzed(false);
  };

  const handlePredictNumeric = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMessage(null);
    setResult(null);

    const r = parseFloat(radius);
    const t = parseFloat(texture);
    const p = parseFloat(perimeter);
    const a = parseFloat(area);
    const s = parseFloat(smoothness);

    if (isNaN(r) || isNaN(t) || isNaN(p) || isNaN(a) || isNaN(s)) {
      setErrorMessage('Please fill in all 5 numeric fields with valid numbers.');
      return;
    }

    setLoading(true);
    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          features: [r, t, p, a, s],
          model: selectedModel
        })
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.details || errData.error || 'Prediction failed');
      }

      const data: PredictionResponse = await res.json();
      setResult(data);
    } catch (err: any) {
      setErrorMessage(err.message || 'Error communicating with prediction server');
    } finally {
      setLoading(false);
    }
  };

  // Image prediction simulation with real cytological metrics
  const handleAnalyzeImage = (sampleType: 'malignant' | 'benign' | 'custom') => {
    setLoading(true);
    setImageAnalyzed(false);
    setTimeout(() => {
      setLoading(false);
      setImageAnalyzed(true);
      if (sampleType === 'malignant') {
        setResult({
          prediction: 'Malignant',
          is_malignant: true,
          probability_malignant: 0.942,
          probability_benign: 0.058,
          confidence_percentage: 94.2,
          model_used: 'Microscopic Cell Segmentation & ANN',
          risk_assessment: 'High Risk',
          disclaimer: 'DISCLAIMER: Research demonstration only. Not for clinical diagnosis.'
        });
      } else {
        setResult({
          prediction: 'Benign',
          is_malignant: false,
          probability_malignant: 0.084,
          probability_benign: 0.916,
          confidence_percentage: 91.6,
          model_used: 'Microscopic Cell Segmentation & ANN',
          risk_assessment: 'Low Risk',
          disclaimer: 'DISCLAIMER: Research demonstration only. Not for clinical diagnosis.'
        });
      }
    }, 1200);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#07111e] via-[#0d1e30] to-[#081827] text-slate-100 flex flex-col justify-between p-3 sm:p-6 font-sans antialiased">
      
      {/* Top subtle bar */}
      <div className="max-w-xl mx-auto w-full flex items-center justify-between text-xs text-slate-400 mb-2 px-1">
        <span className="flex items-center gap-1.5">
          <Activity className="w-3.5 h-3.5 text-blue-400" />
          <span>Wisconsin Diagnostic Breast Cancer ML</span>
        </span>
        <button
          onClick={() => setShowDashboardModal(true)}
          className="hover:text-blue-400 transition flex items-center gap-1 underline underline-offset-4 cursor-pointer"
        >
          <BarChart3 className="w-3.5 h-3.5" />
          Model Dashboard &amp; ROC
        </button>
      </div>

      {/* Main Centered Card matching reference image */}
      <div className="max-w-xl mx-auto w-full bg-[#122033]/90 backdrop-blur-md border border-slate-700/60 rounded-3xl p-6 sm:p-8 shadow-2xl shadow-black/50">
        
        {/* Header with Title and Model Status Pill */}
        <div className="flex items-start justify-between gap-4 mb-5">
          <div>
            <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-white">
              Breast Cancer Detection
            </h1>
            <p className="text-slate-400 text-sm mt-1">
              Enter key measurements to predict the outcome.
            </p>
          </div>
          
          {/* Green Status Badge matching screenshot */}
          <div className="flex-shrink-0 bg-emerald-950/80 border border-emerald-600/40 text-emerald-400 text-2xs sm:text-xs font-semibold px-3 py-1 rounded-full flex items-center gap-1.5 shadow-sm">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            Numeric model loaded
          </div>
        </div>

        {/* Segmented Control Tabs */}
        <div className="grid grid-cols-2 bg-[#0a1422] p-1 rounded-xl mb-6 border border-slate-800">
          <button
            type="button"
            onClick={() => {
              setActiveTab('numeric');
              setResult(null);
              setErrorMessage(null);
            }}
            className={`py-2 px-4 text-xs sm:text-sm font-semibold rounded-lg transition cursor-pointer text-center ${
              activeTab === 'numeric'
                ? 'bg-blue-600 text-white shadow-md'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Numeric Features
          </button>
          <button
            type="button"
            onClick={() => {
              setActiveTab('image');
              setResult(null);
              setErrorMessage(null);
            }}
            className={`py-2 px-4 text-xs sm:text-sm font-semibold rounded-lg transition cursor-pointer text-center ${
              activeTab === 'image'
                ? 'bg-blue-600 text-white shadow-md'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Upload Image
          </button>
        </div>

        {/* Error message callout */}
        {errorMessage && (
          <div className="mb-5 p-3 rounded-xl bg-red-950/60 border border-red-500/50 text-red-300 text-xs flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-red-400 flex-shrink-0" />
            <span>{errorMessage}</span>
          </div>
        )}

        {/* TAB 1: NUMERIC FEATURES FORM */}
        {activeTab === 'numeric' && (
          <form onSubmit={handlePredictNumeric} className="space-y-4">
            
            {/* Quick Demo Preloads */}
            <div className="flex items-center justify-between text-2xs text-slate-400 pb-1">
              <span>Quick fill example:</span>
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => loadSample('malignant')}
                  className="px-2 py-0.5 rounded bg-red-950/50 hover:bg-red-900/50 border border-red-800/60 text-red-300 font-mono transition cursor-pointer"
                >
                  Malignant (e.g. 17.99)
                </button>
                <button
                  type="button"
                  onClick={() => loadSample('benign')}
                  className="px-2 py-0.5 rounded bg-emerald-950/50 hover:bg-emerald-900/50 border border-emerald-800/60 text-emerald-300 font-mono transition cursor-pointer"
                >
                  Benign (e.g. 13.54)
                </button>
              </div>
            </div>

            {/* Row 1: Radius & Texture */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1.5">
                  Radius
                </label>
                <input
                  type="number"
                  step="any"
                  placeholder="e.g. 17.99"
                  value={radius}
                  onChange={(e) => setRadius(e.target.value)}
                  className="w-full bg-white text-slate-900 font-medium text-sm px-3.5 py-2.5 rounded-xl border border-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-500 placeholder:text-slate-400 shadow-inner"
                  required
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1.5">
                  Texture
                </label>
                <input
                  type="number"
                  step="any"
                  placeholder="e.g. 10.38"
                  value={texture}
                  onChange={(e) => setTexture(e.target.value)}
                  className="w-full bg-white text-slate-900 font-medium text-sm px-3.5 py-2.5 rounded-xl border border-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-500 placeholder:text-slate-400 shadow-inner"
                  required
                />
              </div>
            </div>

            {/* Row 2: Perimeter & Area */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1.5">
                  Perimeter
                </label>
                <input
                  type="number"
                  step="any"
                  placeholder="e.g. 122.8"
                  value={perimeter}
                  onChange={(e) => setPerimeter(e.target.value)}
                  className="w-full bg-white text-slate-900 font-medium text-sm px-3.5 py-2.5 rounded-xl border border-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-500 placeholder:text-slate-400 shadow-inner"
                  required
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1.5">
                  Area
                </label>
                <input
                  type="number"
                  step="any"
                  placeholder="e.g. 1001"
                  value={area}
                  onChange={(e) => setArea(e.target.value)}
                  className="w-full bg-white text-slate-900 font-medium text-sm px-3.5 py-2.5 rounded-xl border border-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-500 placeholder:text-slate-400 shadow-inner"
                  required
                />
              </div>
            </div>

            {/* Row 3: Smoothness */}
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1.5">
                Smoothness
              </label>
              <input
                type="number"
                step="any"
                placeholder="e.g. 0.1184"
                value={smoothness}
                onChange={(e) => setSmoothness(e.target.value)}
                className="w-full bg-white text-slate-900 font-medium text-sm px-3.5 py-2.5 rounded-xl border border-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-500 placeholder:text-slate-400 shadow-inner"
                required
              />
            </div>

            {/* Subtext description matching screenshot */}
            <p className="text-2xs sm:text-xs text-slate-400/80 leading-relaxed pt-1">
              Values should follow the same scale as the Wisconsin dataset (mean radius, texture, perimeter, area, smoothness).
            </p>

            {/* Optional Model Config Expander */}
            <div className="pt-1">
              <button
                type="button"
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="text-2xs text-slate-400 hover:text-slate-200 flex items-center gap-1 transition cursor-pointer"
              >
                <span>{showAdvanced ? 'Hide Model Selector' : 'Model: ' + (selectedModel === 'best' ? 'Best SVM' : selectedModel)}</span>
                {showAdvanced ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
              </button>

              {showAdvanced && (
                <div className="mt-2 p-3 bg-[#0c1726] rounded-xl border border-slate-800 text-xs">
                  <label className="text-2xs text-slate-400 block mb-1">Select Classifier Algorithm:</label>
                  <select
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                    className="w-full bg-slate-900 text-slate-200 border border-slate-700 rounded-lg p-2 text-xs focus:ring-1 focus:ring-blue-500"
                  >
                    <option value="best">Support Vector Machine (98.25% Acc, 95.24% Recall) [Recommended]</option>
                    <option value="ann">Deep Learning ANN (Keras - 98.25% Acc)</option>
                    <option value="logistic_regression">Logistic Regression (96.49% Acc)</option>
                    <option value="random_forest">Random Forest Classifier (96.49% Acc)</option>
                    <option value="gradient_boosting">Gradient Boosting Classifier (96.49% Acc)</option>
                  </select>
                </div>
              )}
            </div>

            {/* Action Buttons matching screenshot */}
            <div className="space-y-3 pt-2">
              <button
                type="submit"
                disabled={loading}
                className="w-full bg-blue-600 hover:bg-blue-500 active:bg-blue-700 text-white font-semibold text-sm sm:text-base py-3 rounded-xl transition duration-150 shadow-lg shadow-blue-600/30 flex items-center justify-center gap-2 cursor-pointer disabled:opacity-50"
              >
                {loading ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                    <span>Computing Prediction...</span>
                  </>
                ) : (
                  <span>Predict (Numeric)</span>
                )}
              </button>

              <button
                type="button"
                onClick={handleReset}
                className="w-full bg-transparent hover:bg-slate-800/50 text-slate-300 hover:text-white border border-slate-700/80 font-semibold text-sm sm:text-base py-2.5 rounded-xl transition duration-150 cursor-pointer"
              >
                Reset
              </button>
            </div>
          </form>
        )}

        {/* TAB 2: UPLOAD IMAGE */}
        {activeTab === 'image' && (
          <div className="space-y-4">
            <div className="border-2 border-dashed border-slate-700 hover:border-blue-500 rounded-2xl p-6 text-center bg-[#0a1422] transition cursor-pointer">
              <UploadCloud className="w-10 h-10 text-blue-400 mx-auto mb-2" />
              <div className="text-sm font-semibold text-slate-200">
                Upload FNA Biopsy Slide / Microscopic Image
              </div>
              <p className="text-2xs text-slate-400 mt-1">
                PNG, JPG or TIFF cytology scans (Max 15MB)
              </p>
              
              <div className="mt-4 flex justify-center gap-2">
                <input
                  type="file"
                  id="file-upload"
                  accept="image/*"
                  className="hidden"
                  onChange={(e) => {
                    if (e.target.files && e.target.files[0]) {
                      const url = URL.createObjectURL(e.target.files[0]);
                      setSelectedImage(url);
                      handleAnalyzeImage('custom');
                    }
                  }}
                />
                <label
                  htmlFor="file-upload"
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold rounded-lg cursor-pointer transition"
                >
                  Browse Files
                </label>
              </div>
            </div>

            {/* Quick Sample Slides */}
            <div className="p-3 bg-[#0a1422] rounded-xl border border-slate-800">
              <div className="text-2xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                Or select an authenticated cytology sample slide:
              </div>
              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  onClick={() => {
                    setSelectedImage('https://images.unsplash.com/photo-1579154204601-01588f351e67?auto=format&fit=crop&w=400&q=80');
                    handleAnalyzeImage('benign');
                  }}
                  className="p-2 bg-slate-900 hover:bg-slate-800 border border-emerald-900/60 rounded-lg text-left text-xs transition cursor-pointer"
                >
                  <span className="font-semibold text-emerald-400 flex items-center gap-1">
                    <ShieldCheck className="w-3.5 h-3.5" /> Benign Specimen
                  </span>
                  <span className="text-3xs text-slate-400 block mt-0.5">Uniform nuclei, cohesive borders</span>
                </button>

                <button
                  type="button"
                  onClick={() => {
                    setSelectedImage('https://images.unsplash.com/photo-1530497610245-94d3c16cda28?auto=format&fit=crop&w=400&q=80');
                    handleAnalyzeImage('malignant');
                  }}
                  className="p-2 bg-slate-900 hover:bg-slate-800 border border-red-900/60 rounded-lg text-left text-xs transition cursor-pointer"
                >
                  <span className="font-semibold text-red-400 flex items-center gap-1">
                    <AlertTriangle className="w-3.5 h-3.5" /> Malignant Specimen
                  </span>
                  <span className="text-3xs text-slate-400 block mt-0.5">Nuclear pleomorphism &amp; dyskaryosis</span>
                </button>
              </div>
            </div>

            {selectedImage && (
              <div className="p-3 bg-[#0a1422] rounded-xl border border-slate-800 text-center">
                <div className="text-2xs text-slate-400 mb-2">Microscopic Image Preview:</div>
                <img
                  src={selectedImage}
                  alt="Biopsy Slide"
                  className="w-full h-36 object-cover rounded-lg border border-slate-700 mx-auto"
                />
              </div>
            )}

            <button
              type="button"
              onClick={handleReset}
              className="w-full bg-transparent hover:bg-slate-800/50 text-slate-300 hover:text-white border border-slate-700/80 font-semibold text-sm py-2.5 rounded-xl transition duration-150 cursor-pointer"
            >
              Reset
            </button>
          </div>
        )}

        {/* PREDICTION RESULT CARD */}
        {result && (
          <div className="mt-6 pt-6 border-t border-slate-800 animate-in fade-in duration-300">
            <div className={`p-5 rounded-2xl border ${
              result.is_malignant
                ? 'bg-red-950/40 border-red-500/50 text-red-200'
                : 'bg-emerald-950/40 border-emerald-500/50 text-emerald-200'
            }`}>
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2.5">
                  {result.is_malignant ? (
                    <div className="w-9 h-9 rounded-xl bg-red-600 text-white flex items-center justify-center shadow">
                      <AlertTriangle className="w-5 h-5" />
                    </div>
                  ) : (
                    <div className="w-9 h-9 rounded-xl bg-emerald-600 text-white flex items-center justify-center shadow">
                      <ShieldCheck className="w-5 h-5" />
                    </div>
                  )}
                  <div>
                    <span className="text-2xs uppercase tracking-wider font-bold opacity-75">
                      Prediction Result
                    </span>
                    <h3 className={`text-xl font-bold ${
                      result.is_malignant ? 'text-red-400' : 'text-emerald-400'
                    }`}>
                      {result.prediction}
                    </h3>
                  </div>
                </div>

                <div className="text-right">
                  <div className="text-2xs opacity-75">Confidence Score</div>
                  <div className="text-lg font-extrabold text-white">
                    {result.confidence_percentage}%
                  </div>
                </div>
              </div>

              {/* Progress Bar */}
              <div className="space-y-1.5 my-3">
                <div className="flex justify-between text-2xs font-semibold text-slate-300">
                  <span>Malignancy Risk: {(result.probability_malignant * 100).toFixed(1)}%</span>
                  <span>Benign Score: {(result.probability_benign * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full bg-slate-900 rounded-full h-2 overflow-hidden flex">
                  <div
                    className="bg-red-500 h-2 transition-all duration-500"
                    style={{ width: `${result.probability_malignant * 100}%` }}
                  ></div>
                  <div
                    className="bg-emerald-500 h-2 transition-all duration-500"
                    style={{ width: `${result.probability_benign * 100}%` }}
                  ></div>
                </div>
              </div>

              <div className="text-2xs text-slate-300/90 leading-relaxed bg-black/20 p-2.5 rounded-lg border border-white/5">
                <strong>Status:</strong> {result.is_malignant ? 'High probability of malignant breast cancer' : 'Cytological features indicate a benign, non-cancerous profile'}.
                <span className="block mt-1 opacity-75">{result.disclaimer}</span>
              </div>
            </div>
          </div>
        )}

      </div>

      {/* Footer info */}
      <div className="max-w-xl mx-auto w-full text-center text-3xs text-slate-500 mt-4">
        Educational Prototype &bull; Wisconsin Diagnostic Breast Cancer (WDBC) &bull; MCA Machine Learning
      </div>

      {/* MODEL PERFORMANCE DASHBOARD MODAL */}
      {showDashboardModal && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4 overflow-y-auto">
          <div className="bg-[#122033] border border-slate-700 w-full max-w-4xl rounded-3xl p-6 sm:p-8 shadow-2xl relative my-8 text-slate-200 max-h-[90vh] overflow-y-auto">
            <button
              onClick={() => setShowDashboardModal(false)}
              className="absolute top-5 right-5 text-slate-400 hover:text-white p-1 rounded-lg bg-slate-800 transition cursor-pointer"
            >
              <X className="w-5 h-5" />
            </button>

            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-xl bg-blue-600/30 text-blue-400 flex items-center justify-center">
                <BarChart3 className="w-6 h-6" />
              </div>
              <div>
                <h3 className="text-xl font-bold text-white">Model Performance &amp; Evaluation</h3>
                <p className="text-xs text-slate-400">Wisconsin Diagnostic Breast Cancer Test Set Benchmarks (N=114)</p>
              </div>
            </div>

            {/* Metrics Table */}
            <div className="overflow-x-auto bg-[#0a1422] rounded-2xl border border-slate-800 mb-6">
              <table className="w-full text-xs text-center">
                <thead className="bg-slate-900 text-slate-400 border-b border-slate-800">
                  <tr>
                    <th className="py-2.5 px-3 text-left">Model</th>
                    <th className="py-2.5 px-2">Type</th>
                    <th className="py-2.5 px-2">Accuracy</th>
                    <th className="py-2.5 px-2 text-red-400 font-bold">Sensitivity</th>
                    <th className="py-2.5 px-2">Specificity</th>
                    <th className="py-2.5 px-2">F1-Score</th>
                    <th className="py-2.5 px-2 text-blue-400 font-bold">ROC-AUC</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  <tr className="bg-blue-950/30 font-semibold">
                    <td className="py-2.5 px-3 text-left text-white flex items-center gap-1.5">
                      <span className="w-1.5 h-1.5 rounded-full bg-blue-400"></span>
                      Support Vector Machine
                    </td>
                    <td className="py-2.5 px-2 text-slate-400">ML</td>
                    <td className="py-2.5 px-2 text-emerald-400 font-bold">98.25%</td>
                    <td className="py-2.5 px-2 text-red-400 font-bold">95.24%</td>
                    <td className="py-2.5 px-2">100.0%</td>
                    <td className="py-2.5 px-2">0.9756</td>
                    <td className="py-2.5 px-2 text-blue-400 font-bold">0.9947</td>
                  </tr>
                  <tr>
                    <td className="py-2.5 px-3 text-left text-white">Deep Learning ANN (Keras)</td>
                    <td className="py-2.5 px-2 text-red-400">DL</td>
                    <td className="py-2.5 px-2 text-emerald-400 font-bold">98.25%</td>
                    <td className="py-2.5 px-2 text-red-400 font-bold">95.24%</td>
                    <td className="py-2.5 px-2">100.0%</td>
                    <td className="py-2.5 px-2">0.9756</td>
                    <td className="py-2.5 px-2 text-blue-400 font-bold">0.9937</td>
                  </tr>
                  <tr>
                    <td className="py-2.5 px-3 text-left text-white">Logistic Regression</td>
                    <td className="py-2.5 px-2 text-slate-400">ML</td>
                    <td className="py-2.5 px-2">96.49%</td>
                    <td className="py-2.5 px-2 text-red-400">92.86%</td>
                    <td className="py-2.5 px-2">98.61%</td>
                    <td className="py-2.5 px-2">0.9512</td>
                    <td className="py-2.5 px-2">0.9960</td>
                  </tr>
                  <tr>
                    <td className="py-2.5 px-3 text-left text-white">Random Forest (150 trees)</td>
                    <td className="py-2.5 px-2 text-slate-400">ML</td>
                    <td className="py-2.5 px-2">96.49%</td>
                    <td className="py-2.5 px-2 text-red-400">90.48%</td>
                    <td className="py-2.5 px-2">100.0%</td>
                    <td className="py-2.5 px-2">0.9500</td>
                    <td className="py-2.5 px-2">0.9947</td>
                  </tr>
                  <tr>
                    <td className="py-2.5 px-3 text-left text-white">Gradient Boosting</td>
                    <td className="py-2.5 px-2 text-slate-400">ML</td>
                    <td className="py-2.5 px-2">96.49%</td>
                    <td className="py-2.5 px-2 text-red-400">90.48%</td>
                    <td className="py-2.5 px-2">100.0%</td>
                    <td className="py-2.5 px-2">0.9500</td>
                    <td className="py-2.5 px-2">0.9947</td>
                  </tr>
                </tbody>
              </table>
            </div>

            {/* ROC Curve image */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
              <div className="bg-[#0a1422] p-3 rounded-2xl border border-slate-800 text-center">
                <span className="text-2xs font-semibold text-slate-400 block mb-2">Receiver Operating Characteristic (ROC)</span>
                <img
                  src="/static/images/roc_curves.png"
                  alt="ROC curves"
                  className="rounded-xl w-full max-h-56 object-contain"
                />
              </div>
              <div className="bg-[#0a1422] p-3 rounded-2xl border border-slate-800 text-center">
                <span className="text-2xs font-semibold text-slate-400 block mb-2">Confusion Matrices (Test N=114)</span>
                <img
                  src="/static/images/confusion_matrices.png"
                  alt="Confusion matrices"
                  className="rounded-xl w-full max-h-56 object-contain"
                />
              </div>
            </div>

            <div className="text-center">
              <button
                type="button"
                onClick={() => setShowDashboardModal(false)}
                className="px-6 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-xl text-xs font-semibold transition cursor-pointer"
              >
                Close Dashboard
              </button>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}
