import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useAppStore } from '@/store/useAppStore';
import { useAuthStore } from '@/store/useAuthStore';
import { api, isDemoMode, GoalProgress } from '@/lib/api';
import { Eye, EyeOff, Check, X, Lightbulb, Target } from 'lucide-react';
import { Slider } from '@/components/ui/slider';

export default function SettingsPage() {
  const store = useAppStore();
  const { getUserId } = useAuthStore();
  const [showDeepseek, setShowDeepseek] = useState(false);
  const [showOpenai, setShowOpenai] = useState(false);
  const [testResults, setTestResults] = useState<Record<string, boolean | null>>({
    deepseek: null,
    openai: null,
  });
  const [healthData, setHealthData] = useState<any>(null);

  // Learning Goals state
  const [targetGmat, setTargetGmat] = useState(40);
  const [dailyGoal, setDailyGoal] = useState(5);
  const [goalsSaved, setGoalsSaved] = useState(false);
  const [goalsLoading, setGoalsLoading] = useState(true);
  const [goalData, setGoalData] = useState<GoalProgress | null>(null);

  useEffect(() => {
    const fetchGoals = async () => {
      const userId = getUserId();
      const data = await api.goalProgress(userId);
      setGoalData(data);
      setTargetGmat(data.target_gmat_score);
      setDailyGoal(data.daily_question_goal);
      setGoalsLoading(false);
    };
    fetchGoals();
  }, []);

  const currentGmat = goalData?.current_gmat_score ?? 25;
  const scoreGap = targetGmat - currentGmat;
  const estQuestions = Math.max(scoreGap * 10, 0);
  const estDays = dailyGoal > 0 ? Math.ceil(estQuestions / dailyGoal) : 999;
  const progressPct = ((currentGmat - 20) / (51 - 20)) * 100;
  const targetPct = ((targetGmat - 20) / (51 - 20)) * 100;

  const testConnection = async (key: string) => {
    try {
      const res = await api.health();
      setTestResults((prev) => ({ ...prev, [key]: res.status === 'ok' }));
      setHealthData(res);
    } catch {
      setTestResults((prev) => ({ ...prev, [key]: false }));
    }
  };

  const handleSaveGoals = async () => {
    const userId = getUserId();
    await api.setGoal(userId, targetGmat, dailyGoal);
    setGoalsSaved(true);
    setTimeout(() => setGoalsSaved(false), 3000);
  };

  return (
    <div className="space-y-8 max-w-2xl">
      {isDemoMode() && (
        <div className="px-3 py-2 rounded-lg bg-primary/10 border border-primary/20 text-xs font-mono text-primary flex items-center gap-2">
          <span className="inline-block w-2 h-2 rounded-full bg-primary animate-pulse" />
          DEMO MODE — API Offline · Changes saved locally
        </div>
      )}

      {/* API Configuration */}
      <motion.div
        className="bg-card border border-border rounded-lg p-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h3 className="font-heading text-sm text-foreground tracking-wider mb-5">API CONFIGURATION</h3>
        {[
          { label: 'DeepSeek API Key', value: store.deepseekApiKey, set: store.setDeepseekApiKey, show: showDeepseek, toggle: () => setShowDeepseek(!showDeepseek), testKey: 'deepseek' },
          { label: 'OpenAI API Key', value: store.openaiApiKey, set: store.setOpenaiApiKey, show: showOpenai, toggle: () => setShowOpenai(!showOpenai), testKey: 'openai' },
        ].map((field) => (
          <div key={field.label} className="mb-4">
            <label className="text-[10px] font-mono text-foreground/80 uppercase tracking-widest mb-1.5 block">{field.label}</label>
            <div className="flex gap-2">
              <div className="relative flex-1">
                <input
                  type={field.show ? 'text' : 'password'}
                  value={field.value}
                  onChange={(e) => field.set(e.target.value)}
                  placeholder="sk-..."
                  className="w-full bg-muted border border-border rounded px-3 py-2 text-sm font-mono text-foreground placeholder:text-muted-foreground focus:outline-none focus:border-primary/50"
                />
                <button onClick={field.toggle} className="absolute right-2 top-1/2 -translate-y-1/2 text-foreground/70 hover:text-foreground">
                  {field.show ? <EyeOff size={14} /> : <Eye size={14} />}
                </button>
              </div>
              <button
                onClick={() => testConnection(field.testKey)}
                className="px-3 py-2 bg-muted border border-border rounded text-xs font-mono text-foreground/80 hover:text-foreground hover:border-primary/30 transition-colors"
              >
                Test
              </button>
              {testResults[field.testKey] !== null && (
                <span className={`flex items-center ${testResults[field.testKey] ? 'text-success' : 'text-destructive'}`}>
                  {testResults[field.testKey] ? <Check size={16} /> : <X size={16} />}
                </span>
              )}
            </div>
          </div>
        ))}
      </motion.div>

      {/* Display */}
      <motion.div
        className="bg-card border border-border rounded-lg p-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <h3 className="font-heading text-sm text-foreground tracking-wider mb-5">DISPLAY</h3>
        <div>
          <label className="text-[10px] font-mono text-foreground/80 uppercase tracking-widest mb-2 block">Animation Intensity</label>
          <div className="flex gap-2">
            {(['off', 'subtle', 'full'] as const).map((level) => (
              <button
                key={level}
                onClick={() => store.setAnimationIntensity(level)}
                className={`px-4 py-2 rounded text-xs font-mono uppercase tracking-wider border transition-all ${
                  store.animationIntensity === level
                    ? 'bg-primary/10 border-primary/30 text-primary'
                    : 'bg-muted border-border text-foreground/70 hover:border-primary/20'
                }`}
              >
                {level}
              </button>
            ))}
          </div>
        </div>
      </motion.div>

      {/* Learning Goals */}
      <motion.div
        className="bg-card border border-border rounded-lg p-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.15 }}
      >
        <h3 className="font-heading text-sm text-foreground tracking-wider mb-5 flex items-center gap-2">
          <Target size={16} className="text-primary" /> LEARNING GOALS ⚡
        </h3>

        {goalsLoading ? (
          <p className="font-mono text-sm text-muted-foreground">Loading goals...</p>
        ) : (
          <>
            {/* Target GMAT Score */}
            <div className="mb-6">
              <label className="text-[10px] font-mono text-foreground/80 uppercase tracking-widest mb-2 block">
                Target GMAT Score: <span className="text-primary font-bold">V{targetGmat}</span>
              </label>
              <Slider value={[targetGmat]} onValueChange={(v) => setTargetGmat(v[0])} min={20} max={51} step={1} className="mb-1" />
              <div className="flex justify-between text-[9px] font-mono text-muted-foreground">
                <span>V20</span><span>V51</span>
              </div>
            </div>

            {/* Daily Question Goal */}
            <div className="mb-6">
              <label className="text-[10px] font-mono text-foreground/80 uppercase tracking-widest mb-2 block">
                Daily Question Goal: <span className="text-primary font-bold">{dailyGoal} questions/day</span>
              </label>
              <Slider value={[dailyGoal]} onValueChange={(v) => setDailyGoal(v[0])} min={1} max={20} step={1} className="mb-1" />
              <div className="flex justify-between text-[9px] font-mono text-muted-foreground">
                <span>1</span><span>20</span>
              </div>
            </div>

            {/* Save button */}
            <button
              onClick={handleSaveGoals}
              className="w-full py-3 rounded-lg rainbow-btn font-heading text-sm uppercase tracking-widest glow-hover mb-4"
            >
              {goalsSaved ? 'GOALS UPDATED ✅' : 'Save Goals ⚡'}
            </button>

            {/* Progress visualization */}
            <div className="border border-border rounded-lg p-4" style={{ backgroundColor: 'hsl(270, 100%, 8%)' }}>
              <div className="relative h-4 bg-muted rounded-full overflow-visible mb-2">
                <div
                  className="absolute h-full rounded-full"
                  style={{ width: `${progressPct}%`, background: 'linear-gradient(90deg, hsl(56, 100%, 50%), hsl(45, 100%, 48%))' }}
                />
                <div
                  className="absolute top-1/2 w-4 h-4 rounded-full bg-primary border-2 border-primary-foreground z-10"
                  style={{ left: `${progressPct}%`, transform: 'translate(-50%, -50%)' }}
                />
                <div
                  className="absolute top-1/2 w-4 h-4 rounded-full border-2 border-secondary bg-transparent z-10"
                  style={{ left: `${targetPct}%`, transform: 'translate(-50%, -50%)' }}
                />
              </div>
              <div className="flex justify-between text-[9px] font-mono text-muted-foreground mb-3">
                <span>V20</span><span>V51</span>
              </div>
              <div className="space-y-1 text-xs font-mono text-muted-foreground">
                <p>Current: <span className="text-primary">V{currentGmat}</span> → Target: <span className="text-secondary">V{targetGmat}</span> — {scoreGap > 0 ? `${scoreGap} points to go` : 'Goal reached!'}</p>
                <p>Estimated ~{estQuestions} questions remaining</p>
                <p>At {dailyGoal} questions/day → ~{estDays} days to reach your goal</p>
              </div>
            </div>
          </>
        )}
      </motion.div>

      {/* System Health */}
      <motion.div
        className="bg-card border border-border rounded-lg p-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <div className="flex items-center justify-between mb-5">
          <h3 className="font-heading text-sm text-foreground tracking-wider">SYSTEM HEALTH</h3>
          <button onClick={() => testConnection('deepseek')} className="text-[10px] font-mono text-foreground/70 hover:text-primary transition-colors">Refresh</button>
        </div>
        {healthData ? (
          <div className="space-y-2">
            {[
              { label: 'API Server', ok: healthData.status === 'ok' },
              { label: 'Database', ok: healthData.db_status === 'connected' },
              { label: 'Qdrant Vector DB', ok: healthData.qdrant_status === 'connected' },
            ].map((s) => (
              <div key={s.label} className="flex items-center gap-2 text-sm font-mono">
                <div className={`w-2 h-2 rounded-full ${s.ok ? 'bg-success' : 'bg-destructive'}`} />
                <span className="text-foreground/80">{s.label}</span>
                <span className={`ml-auto text-xs ${s.ok ? 'text-success' : 'text-destructive'}`}>{s.ok ? 'Connected' : 'Disconnected'}</span>
              </div>
            ))}
          </div>
        ) : (
          <p className="font-mono text-sm text-foreground/80">Click refresh to check system status.</p>
        )}
      </motion.div>

      {/* About */}
      <motion.div
        className="bg-card border border-border rounded-lg p-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
      >
        <div className="flex items-center gap-3 mb-3">
          <Lightbulb size={24} className="text-primary" />
          <div>
            <h3 className="font-heading text-sm text-primary tracking-wider">GLITCHMIND LABS</h3>
            <span className="text-[10px] font-mono text-foreground/70">v1.0.0</span>
          </div>
        </div>
        <p className="font-mono text-xs text-foreground/70 italic">"Fix the glitch. Master the logic."</p>
      </motion.div>
    </div>
  );
}
