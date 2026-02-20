import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { AnimatedNumber } from '@/components/AnimatedNumber';
import { Flame, TrendingUp, Target, Brain, AlertTriangle, RotateCcw, ArrowRight, Lightbulb } from 'lucide-react';
import { useAppStore } from '@/store/useAppStore';
import { useAuthStore } from '@/store/useAuthStore';
import { api, isDemoMode, DashboardSummary } from '@/lib/api';

export default function Dashboard() {
  const navigate = useNavigate();
  const store = useAppStore();
  const { user, getUserId } = useAuthStore();
  const [data, setData] = useState<DashboardSummary | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchDashboard = async () => {
      try {
        const res = await api.dashboardSummary(getUserId());
        setData(res);
      } catch {
        // fallback handled by api layer
      } finally {
        setLoading(false);
      }
    };
    fetchDashboard();
  }, []);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-[60vh] gap-6 yellow-atmosphere">
        <div className="loading-bulb-container">
          <Lightbulb size={64} className="loading-bulb-icon loading-bulb-fill" stroke="hsl(0, 0%, 0%)" strokeWidth={1.5} />
        </div>
        <p className="font-mono text-lg tracking-wider font-bold chromatic-text" style={{ color: 'hsl(0, 0%, 0%)' }}>Loading dashboard...</p>
      </div>
    );
  }

  const d = data!;
  const streakDays = d.streak_days;
  const progressPct = d.today_goal > 0 ? Math.min((d.today_completed / d.today_goal) * 100, 100) : 0;
  const onTrack = d.today_completed >= d.today_goal;
  const last7 = d.last_7_days || [false, false, false, false, false, false, false];
  const dayLabels = ['M', 'T', 'W', 'T', 'F', 'S', 'S'];
  const displayName = user?.display_name || 'User';

  return (
    <div className="space-y-6 yellow-atmosphere">
      {isDemoMode() && (
        <div className="px-3 py-2 rounded-lg bg-primary/10 border border-primary/20 text-xs font-mono text-primary flex items-center gap-2">
          <span className="inline-block w-2 h-2 rounded-full bg-primary animate-pulse" />
          DEMO MODE — API Offline · Using sample data
        </div>
      )}

      {/* Welcome Banner */}
      <motion.div
        className="flex items-center gap-4 flex-wrap"
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h1 className="font-heading text-2xl font-bold tracking-wider">
          <span className="tab-rainbow-text" style={{ color: 'hsl(0, 0%, 0%)' }}>Welcome back, {displayName} ⚡</span>
        </h1>
        <span className="px-3 py-1 rounded-full text-sm font-mono font-bold bg-card border border-border text-foreground">
          Day {streakDays} streak 🔥
        </span>
        {streakDays >= 7 && (
          <span className="font-heading text-sm font-bold rainbow-text chromatic-text animate-pulse">
            UNSTOPPABLE 🔥🔥🔥
          </span>
        )}
      </motion.div>

      {/* Today's Progress */}
      <motion.div
        className="bg-card border border-border rounded-lg p-5"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.05 }}
      >
        <h3 className="font-heading text-sm text-foreground tracking-wider mb-3">TODAY'S PROGRESS</h3>
        <div className="flex items-center gap-4">
          <div className="flex-1">
            <div className="flex justify-between text-xs font-mono text-muted-foreground mb-2">
              <span>{d.today_completed} / {d.today_goal} questions today</span>
              <span className={`font-bold ${onTrack ? 'text-success' : 'text-warning'}`}>
                {onTrack ? 'ON TRACK ✅' : 'KEEP GOING ⚠️'}
              </span>
            </div>
            <div className="h-3 bg-muted rounded-full overflow-hidden">
              <motion.div
                className={`h-full rounded-full ${onTrack ? 'shadow-[0_0_12px_hsl(56,100%,50%,0.5)]' : ''}`}
                style={{ background: 'linear-gradient(90deg, hsl(56, 100%, 50%), hsl(45, 100%, 48%))' }}
                initial={{ width: 0 }}
                animate={{ width: `${progressPct}%` }}
                transition={{ duration: 0.8 }}
              />
            </div>
          </div>
          <button
            onClick={() => navigate('/practice')}
            className="px-4 py-2.5 rounded-lg rainbow-btn font-heading text-xs uppercase tracking-widest glow-hover flex items-center gap-2 flex-shrink-0"
          >
            Start Practicing <ArrowRight size={14} />
          </button>
        </div>
      </motion.div>

      {/* 3 Stat Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* GMAT Score */}
        <motion.div className="bg-card border border-border rounded-lg p-5" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
          <div className="flex items-center gap-2 mb-3">
            <TrendingUp size={14} className="text-muted-foreground" />
            <span className="text-[10px] font-mono text-muted-foreground uppercase tracking-widest">GMAT Score</span>
          </div>
          <div className="text-3xl font-heading font-bold text-primary mb-2">V{d.gmat_score}</div>
          <div className="h-2 bg-muted rounded-full overflow-hidden mb-1">
            <div className="h-full rounded-full bg-primary" style={{ width: `${((d.gmat_score - 20) / 31) * 100}%` }} />
          </div>
          <div className="flex justify-between text-[9px] font-mono text-muted-foreground">
            <span>V20</span><span>V51</span>
          </div>
          <p className="text-xs font-mono text-secondary mt-2">θ {d.current_theta.toFixed(2)}</p>
        </motion.div>

        {/* Accuracy */}
        <motion.div className="bg-card border border-border rounded-lg p-5" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }}>
          <div className="flex items-center gap-2 mb-3">
            <Target size={14} className="text-muted-foreground" />
            <span className="text-[10px] font-mono text-muted-foreground uppercase tracking-widest">Accuracy</span>
          </div>
          <div className="flex items-center justify-center my-2">
            <div className="relative w-24 h-24">
              <svg className="w-full h-full -rotate-90" viewBox="0 0 36 36">
                <circle cx="18" cy="18" r="15.5" fill="none" stroke="hsl(260, 40%, 14%)" strokeWidth="3" />
                <circle cx="18" cy="18" r="15.5" fill="none" stroke="hsl(56, 100%, 50%)" strokeWidth="3" strokeDasharray={`${d.accuracy_pct} ${100 - d.accuracy_pct}`} strokeLinecap="round" />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-xl font-heading font-bold text-primary">{d.accuracy_pct}%</span>
              </div>
            </div>
          </div>
          <p className="text-xs font-mono text-muted-foreground text-center">based on {d.total_questions} questions</p>
        </motion.div>

        {/* Streak */}
        <motion.div className="bg-card border border-border rounded-lg p-5" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
          <div className="flex items-center gap-2 mb-3">
            <Flame size={14} className="text-muted-foreground" />
            <span className="text-[10px] font-mono text-muted-foreground uppercase tracking-widest">Streak</span>
          </div>
          <div className="text-3xl font-heading font-bold text-warning mb-1 flex items-center gap-2">
            {streakDays} <span className="text-2xl">🔥</span>
          </div>
          <p className="text-xs font-mono text-muted-foreground mb-3">consecutive days</p>
          <div className="flex gap-1.5">
            {last7.map((practiced, i) => (
              <div key={i} className="flex flex-col items-center gap-1">
                <div className={`w-5 h-5 rounded-sm ${practiced ? 'bg-success' : 'border border-border bg-transparent'}`} />
                <span className="text-[8px] font-mono text-muted-foreground">{dayLabels[i]}</span>
              </div>
            ))}
          </div>
        </motion.div>
      </div>

      {/* Two cards side by side */}
      <div className="grid lg:grid-cols-2 gap-4">
        {/* Weak Skills */}
        <motion.div className="bg-card border border-border rounded-lg p-5" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.25 }}>
          <h3 className="font-heading text-sm text-foreground tracking-wider mb-4 flex items-center gap-2">
            <AlertTriangle size={14} className="text-warning" /> WEAK SKILLS ⚠️
          </h3>
          {d.weak_skills.length === 0 ? (
            <p className="text-xs font-mono text-muted-foreground">Start practicing to see your weak skills.</p>
          ) : (
            <div className="space-y-3">
              {d.weak_skills.map((skill) => (
                <div key={skill.skill_name}>
                  <div className="flex justify-between text-xs font-mono mb-1">
                    <span className="text-foreground">{skill.skill_name}</span>
                    <span className="text-destructive">{Math.round(skill.error_rate * 100)}%</span>
                  </div>
                  <div className="h-2 bg-muted rounded-full overflow-hidden">
                    <div className="h-full rounded-full bg-destructive/70" style={{ width: `${skill.error_rate * 100}%` }} />
                  </div>
                </div>
              ))}
            </div>
          )}
          <button
            onClick={() => navigate('/practice')}
            className="mt-4 w-full py-2 rounded-lg border border-primary/30 text-xs font-mono text-primary hover:bg-primary/10 transition-colors flex items-center justify-center gap-2"
          >
            Practice Weak Skills <ArrowRight size={12} />
          </button>
        </motion.div>

        {/* Reviews Due */}
        <motion.div className="bg-card border border-border rounded-lg p-5" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}>
          <h3 className="font-heading text-sm text-foreground tracking-wider mb-4 flex items-center gap-2">
            <RotateCcw size={14} className="text-secondary" /> REVIEWS DUE 🔄
          </h3>
          <div className="flex flex-col items-center py-4">
            <span className={`text-5xl font-heading font-bold ${
              d.reviews_due > 5 ? 'text-destructive' : d.reviews_due > 0 ? 'text-warning' : 'text-success'
            }`}>
              {d.reviews_due}
            </span>
            <p className="text-xs font-mono text-muted-foreground mt-2">questions losing signal</p>
          </div>
          <button
            onClick={() => navigate('/analytics', { state: { tab: 'review' } })}
            className="w-full py-2 rounded-lg border border-secondary/30 text-xs font-mono text-secondary hover:bg-secondary/10 transition-colors flex items-center justify-center gap-2"
          >
            Go to Review <ArrowRight size={12} />
          </button>
        </motion.div>
      </div>
    </div>
  );
}
