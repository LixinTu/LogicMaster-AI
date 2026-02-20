import { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { useAppStore } from '@/store/useAppStore';
import { useAuthStore } from '@/store/useAuthStore';
import { AnimatedNumber } from '@/components/AnimatedNumber';
import { motion } from 'framer-motion';
import { TrendingUp, Target, Brain, Flame, Zap, Lightbulb } from 'lucide-react';
import {
  LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip,
  RadarChart, PolarGrid, PolarAngleAxis, Radar,
  PieChart, Pie, Cell, BarChart, Bar,
} from 'recharts';
import { api, ReviewItem, isDemoMode, AnalyticsData, GoalProgress } from '@/lib/api';

function AnalyticsTab() {
  const { totalAnswered, accuracy, theta, streak, questionsLog } = useAppStore();
  const { getUserId } = useAuthStore();
  const [analyticsData, setAnalyticsData] = useState<AnalyticsData | null>(null);
  const [goalData, setGoalData] = useState<GoalProgress | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      const userId = getUserId();
      const [ad, gd] = await Promise.all([
        api.analyticsData(userId),
        api.goalProgress(userId),
      ]);
      setAnalyticsData(ad);
      setGoalData(gd);
      setLoading(false);
    };
    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-[40vh] gap-6">
        <div className="loading-bulb-container">
          <Lightbulb size={64} className="loading-bulb-icon loading-bulb-fill" stroke="hsl(0, 0%, 0%)" strokeWidth={1.5} />
        </div>
        <p className="font-mono text-lg tracking-wider font-bold chromatic-text" style={{ color: 'hsl(0, 0%, 0%)' }}>Loading analytics...</p>
      </div>
    );
  }

  // Use API data if available, otherwise fall back to local Zustand store data
  const answerHistory = analyticsData?.answer_history || questionsLog.map(l => ({ question_id: l.question_id, is_correct: l.is_correct, theta_at_time: l.theta_at_time }));
  const curveData = answerHistory.length > 0
    ? answerHistory.map((h, i) => ({ question: i + 1, theta: h.theta_at_time }))
    : [];
  const skillData = analyticsData?.skill_mastery || [];
  const wrongByType = analyticsData?.wrong_by_type || [];
  const wrongBySkill = analyticsData?.wrong_by_skill || [];
  const totalWrong = wrongByType.reduce((s, t) => s + t.value, 0);

  const g = goalData;
  const dayLabels = ['M', 'T', 'W', 'T', 'F', 'S', 'S'];

  const cards = [
    { label: 'Questions', value: totalAnswered, icon: Brain, color: 'text-secondary' },
    { label: 'Accuracy', value: accuracy, suffix: '%', icon: Target, color: 'text-primary' },
    { label: 'Theta', value: theta, decimals: 2, icon: TrendingUp, color: 'text-secondary' },
    { label: 'Best Streak', value: streak, icon: Flame, color: 'text-warning' },
  ];

  const hasData = answerHistory.length > 0 || totalAnswered > 0;

  return (
    <div className="space-y-6">
      {isDemoMode() && (
        <div className="px-3 py-2 rounded-lg bg-primary/10 border border-primary/20 text-xs font-mono text-primary flex items-center gap-2">
          <span className="inline-block w-2 h-2 rounded-full bg-primary animate-pulse" />
          DEMO MODE — API Offline · Showing local data
        </div>
      )}

      {/* Goal Progress */}
      {g && (
        <motion.div className="bg-card border border-border rounded-lg p-5" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          <h3 className="font-heading text-sm text-foreground tracking-wider mb-4 flex items-center gap-2">
            GOAL PROGRESS
            <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold text-secondary border border-secondary/30 bg-secondary/10">GOAL: V{g.target_gmat_score}</span>
          </h3>
          <div className="relative h-4 bg-muted rounded-full overflow-visible mb-2">
            <div
              className="absolute h-full rounded-full"
              style={{ width: `${((g.current_gmat_score - 20) / 31) * 100}%`, background: 'linear-gradient(90deg, hsl(56, 100%, 50%), hsl(45, 100%, 48%))' }}
            />
            <div
              className="absolute top-1/2 -translate-y-1/2 w-4 h-4 rounded-full bg-primary border-2 border-primary-foreground z-10"
              style={{ left: `${((g.current_gmat_score - 20) / 31) * 100}%`, transform: `translate(-50%, -50%)` }}
            />
            <div
              className="absolute top-1/2 -translate-y-1/2 w-4 h-4 rounded-full border-2 border-secondary bg-transparent z-10"
              style={{ left: `${((g.target_gmat_score - 20) / 31) * 100}%`, transform: `translate(-50%, -50%)` }}
            />
          </div>
          <div className="flex justify-between text-[9px] font-mono text-muted-foreground mb-3">
            <span>V20</span><span>V51</span>
          </div>
          <p className="text-xs font-mono text-muted-foreground mb-3">
            {g.target_gmat_score > g.current_gmat_score
              ? `${Math.round(((g.current_gmat_score - 20) / (g.target_gmat_score - 20)) * 100)}% of the way there`
              : 'Goal reached! 🎉'}
          </p>
          <div className="flex gap-1.5">
            {(g.last_7_days || []).map((met, i) => (
              <div key={i} className="flex flex-col items-center gap-1">
                <div className={`w-5 h-5 rounded-sm ${met ? 'bg-success' : 'border border-border bg-transparent'}`} />
                <span className="text-[8px] font-mono text-muted-foreground">{dayLabels[i]}</span>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Metric cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {cards.map((card, i) => (
          <motion.div key={card.label} className="bg-card border border-border rounded-lg p-4" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.05 }}>
            <div className="flex items-center gap-2 mb-2">
              <card.icon size={14} className="text-foreground/80" />
              <span className="text-[10px] font-mono text-foreground/80 uppercase tracking-widest">{card.label}</span>
            </div>
            <AnimatedNumber value={card.value} decimals={card.decimals || 0} suffix={card.suffix || ''} className={`text-2xl font-heading font-bold ${card.color}`} />
          </motion.div>
        ))}
      </div>

      {!hasData ? (
        <motion.div className="flex flex-col items-center justify-center h-[30vh] gap-4" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          <Lightbulb size={48} className="text-primary/50" />
          <p className="font-mono text-sm text-muted-foreground">Start practicing to see your data.</p>
        </motion.div>
      ) : (
        <>
          <div className="grid lg:grid-cols-2 gap-6">
            {/* Learning Curve */}
            {curveData.length > 0 && (
              <motion.div className="bg-card border border-border rounded-lg p-5" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}>
                <h3 className="font-heading text-sm text-foreground tracking-wider mb-4">LEARNING CURVE</h3>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={curveData}>
                    <XAxis dataKey="question" tick={{ fontSize: 10, fill: 'hsl(0, 0%, 75%)' }} />
                    <YAxis tick={{ fontSize: 10, fill: 'hsl(0, 0%, 75%)' }} domain={['auto', 'auto']} />
                    <Tooltip contentStyle={{ backgroundColor: 'hsl(240, 17%, 8%)', border: '1px solid hsl(56, 100%, 50%, 0.1)', borderRadius: '8px', fontSize: '12px', fontFamily: 'JetBrains Mono' }} />
                    <Line type="monotone" dataKey="theta" stroke="hsl(56, 100%, 50%)" strokeWidth={2} dot={{ fill: 'hsl(180, 100%, 50%)', r: 3 }} />
                  </LineChart>
                </ResponsiveContainer>
              </motion.div>
            )}

            {/* Skill Radar */}
            {skillData.length > 0 && (
              <motion.div className="bg-card border border-border rounded-lg p-5" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }}>
                <h3 className="font-heading text-sm text-foreground tracking-wider mb-4">SKILL MASTERY</h3>
                <ResponsiveContainer width="100%" height={200}>
                  <RadarChart data={skillData}>
                    <PolarGrid stroke="hsl(0, 0%, 30%)" />
                    <PolarAngleAxis dataKey="skill" tick={{ fontSize: 9, fill: 'hsl(0, 0%, 75%)' }} />
                    <Radar dataKey="value" stroke="hsl(180, 100%, 50%)" fill="hsl(56, 100%, 50%)" fillOpacity={0.15} strokeWidth={2} />
                  </RadarChart>
                </ResponsiveContainer>
              </motion.div>
            )}
          </div>

          {/* Wrong Answer Analysis */}
          {(wrongByType.length > 0 || wrongBySkill.length > 0) && (
            <div className="grid lg:grid-cols-2 gap-6">
              {wrongByType.length > 0 && (
                <motion.div className="bg-card border border-border rounded-lg p-5" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.4 }}>
                  <h3 className="font-heading text-sm text-foreground tracking-wider mb-4">WRONG BY TYPE</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart margin={{ top: 30, right: 30, bottom: 30, left: 30 }}>
                      <Pie data={wrongByType} cx="50%" cy="50%" innerRadius={50} outerRadius={80} dataKey="value" label={({ name, value }) => `${name} (${value})`} labelLine={true} fontSize={10}>
                        {wrongByType.map((entry, i) => (
                          <Cell key={i} fill={entry.color} />
                        ))}
                      </Pie>
                    </PieChart>
                  </ResponsiveContainer>
                  <p className="text-center text-xs font-mono text-muted-foreground mt-1">Total: {totalWrong} wrong</p>
                </motion.div>
              )}

              {wrongBySkill.length > 0 && (
                <motion.div className="bg-card border border-border rounded-lg p-5" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.45 }}>
                  <h3 className="font-heading text-sm text-foreground tracking-wider mb-4">WRONG BY SKILL</h3>
                  <ResponsiveContainer width="100%" height={280}>
                    <BarChart data={wrongBySkill} layout="vertical" margin={{ top: 30, right: 20, bottom: 10, left: 10 }}>
                      <XAxis type="number" tick={{ fontSize: 10, fill: 'hsl(0, 0%, 75%)' }} />
                      <YAxis type="category" dataKey="skill" tick={{ fontSize: 9, fill: 'hsl(0, 0%, 75%)' }} width={95} />
                      <Tooltip contentStyle={{ backgroundColor: 'hsl(240, 17%, 8%)', border: '1px solid hsl(260, 60%, 30%, 0.3)', borderRadius: '8px', fontSize: '12px', fontFamily: 'JetBrains Mono' }} />
                      <Bar dataKey="count" fill="hsl(345, 100%, 60%)" radius={[0, 4, 4, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </motion.div>
              )}
            </div>
          )}

          {/* Session History */}
          {answerHistory.length > 0 && (
            <motion.div className="bg-card border border-border rounded-lg p-5" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }}>
              <h3 className="font-heading text-sm text-foreground tracking-wider mb-4">SESSION LOG</h3>
              <div className="space-y-1">
                <div className="grid grid-cols-4 text-[10px] font-mono text-foreground/70 uppercase tracking-widest pb-2 border-b border-border">
                  <span>#</span><span>Question ID</span><span>Result</span><span>Theta</span>
                </div>
                {answerHistory.slice(-20).reverse().map((log, i) => (
                  <div key={i} className="grid grid-cols-4 text-xs font-mono py-1.5 text-foreground/80 border-b border-border/30">
                    <span>{answerHistory.length - i}</span>
                    <span className="truncate">{log.question_id.slice(0, 8)}...</span>
                    <span className={log.is_correct ? 'text-success' : 'text-destructive'}>{log.is_correct ? '✓ PASS' : '✗ FAIL'}</span>
                    <span>{log.theta_at_time.toFixed(2)}</span>
                  </div>
                ))}
              </div>
            </motion.div>
          )}
        </>
      )}
    </div>
  );
}

function ReviewTab() {
  const navigate = useNavigate();
  const { getUserId } = useAuthStore();
  const [reviews, setReviews] = useState<ReviewItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [dueCount, setDueCount] = useState(0);

  useEffect(() => {
    const fetchReviews = async () => {
      try {
        const res = await api.reviewSchedule(getUserId(), 0.5);
        setReviews(res.reviews);
        setDueCount(res.due_count);
      } catch (err) {
        console.error('Failed to fetch reviews:', err);
      } finally {
        setLoading(false);
      }
    };
    fetchReviews();
  }, []);

  const getRecallInfo = (prob: number) => {
    if (prob > 0.5) return { color: 'hsl(152, 100%, 50%)', label: 'STRONG', textClass: 'text-success' };
    if (prob >= 0.3) return { color: 'hsl(56, 100%, 50%)', label: 'FADING', textClass: 'text-primary' };
    return { color: 'hsl(345, 100%, 60%)', label: 'CRITICAL — REVIEW NOW', textClass: 'text-destructive' };
  };

  const getTypeBadgeClass = (type?: string) => {
    switch (type?.toLowerCase()) {
      case 'weaken': return 'bg-destructive/20 text-destructive border-destructive/30';
      case 'strengthen': return 'bg-success/20 text-success border-success/30';
      case 'assumption': return 'bg-primary/20 text-primary border-primary/30';
      case 'inference': return 'bg-secondary/20 text-secondary border-secondary/30';
      default: return 'bg-muted text-muted-foreground border-border';
    }
  };
  const getDifficultyBadgeClass = (diff?: string) => {
    switch (diff?.toLowerCase()) {
      case 'hard': return 'bg-destructive/10 text-destructive/80 border-destructive/20';
      case 'medium': return 'bg-warning/10 text-warning/80 border-warning/20';
      case 'easy': return 'bg-success/10 text-success/80 border-success/20';
      default: return 'bg-muted text-muted-foreground border-border';
    }
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-[40vh] gap-6">
        <div className="loading-bulb-container">
          <Lightbulb size={64} className="loading-bulb-icon loading-bulb-fill" stroke="hsl(0, 0%, 0%)" strokeWidth={1.5} />
        </div>
        <p className="font-mono text-lg tracking-wider font-bold chromatic-text" style={{ color: 'hsl(0, 0%, 0%)' }}>Scanning memory banks...</p>
      </div>
    );
  }

  if (reviews.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-[40vh] gap-6">
        <div className="loading-bulb-container">
          <Lightbulb size={64} className="loading-bulb-icon loading-bulb-fill" stroke="hsl(0, 0%, 0%)" strokeWidth={1.5} />
        </div>
        <p className="font-mono text-lg tracking-wider font-bold chromatic-text" style={{ color: 'hsl(0, 0%, 0%)' }}>All systems nominal.</p>
        <p className="font-mono text-sm text-muted-foreground text-center max-w-md">No bugs detected in your memory banks.</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="font-mono text-xl font-bold tracking-wide uppercase text-primary-foreground chromatic-text">MEMORY MAINTENANCE</h2>
        <span className="px-4 py-1.5 rounded-full text-sm font-mono border border-border" style={{ backgroundColor: 'hsl(270, 100%, 8%)' }}>
          <span className="rainbow-logo-text">{dueCount} items need review</span>
        </span>
      </div>
      <div className="space-y-3">
        {reviews.map((item, i) => {
          const recall = getRecallInfo(item.recall_probability);
          return (
            <motion.div key={item.question_id} className="bg-card border border-border rounded-lg p-4" initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.05 }}>
              <div className="flex items-start justify-between gap-4">
                <div className="flex-1 min-w-0 space-y-3">
                  <div className="flex flex-wrap items-center gap-2">
                    {item.question_type && <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase border ${getTypeBadgeClass(item.question_type)}`}>{item.question_type}</span>}
                    {item.difficulty && <span className={`px-2 py-0.5 rounded text-[10px] font-mono uppercase border ${getDifficultyBadgeClass(item.difficulty)}`}>{item.difficulty}</span>}
                    {item.skills?.map((skill, si) => <span key={si} className={`px-2 py-0.5 rounded text-[10px] font-mono border skill-badge-${si % 5}`}>{skill}</span>)}
                  </div>
                  {item.stimulus_preview && <p className="text-xs font-mono text-foreground/90 truncate">{item.stimulus_preview}</p>}
                  <div>
                    <div className="flex justify-between text-[10px] font-mono mb-1">
                      <span className={recall.textClass} style={{ fontWeight: 700 }}>{recall.label}</span>
                      <span className="text-foreground/90">{Math.round(item.recall_probability * 100)}%</span>
                    </div>
                    <div className="h-2 bg-muted rounded-full overflow-hidden">
                      <motion.div className="h-full rounded-full" style={{ backgroundColor: recall.color }} initial={{ width: 0 }} animate={{ width: `${item.recall_probability * 100}%` }} transition={{ duration: 0.5, delay: i * 0.05 }} />
                    </div>
                  </div>
                  <div className="flex gap-4 text-[10px] font-mono text-foreground/80">
                    <span>Memory half-life: {item.half_life.toFixed(1)} days</span>
                    <span>Last practiced: {item.elapsed_days.toFixed(1)} days ago</span>
                  </div>
                </div>
                <button
                  onClick={() => navigate('/practice', { state: { reviewQuestionId: item.question_id } })}
                  className="px-3 py-2.5 rounded-md text-xs font-mono tracking-wide uppercase sidebar-rainbow-bg text-primary-foreground font-bold shadow-[0_0_24px_hsl(56,100%,50%,0.5)] chromatic-text flex items-center gap-1.5 flex-shrink-0"
                >
                  <Zap size={12} /> REVIEW NOW
                </button>
              </div>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}

export default function Analytics() {
  const location = useLocation();
  const initialTab = (location.state as any)?.tab === 'review' ? 'review' : 'analytics';
  const [activeTab, setActiveTab] = useState(initialTab);

  useEffect(() => {
    const state = location.state as any;
    if (state?.tab === 'review') {
      setActiveTab('review');
      window.history.replaceState({}, '');
    }
  }, [location.state]);

  return (
    <div className="space-y-6 yellow-atmosphere">
      <div className="flex gap-1 border-b border-border">
        {[
          { key: 'analytics', label: '📊 ANALYTICS' },
          { key: 'review', label: '🔄 REVIEW' },
        ].map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className={`px-5 py-3 font-heading text-xs uppercase tracking-widest transition-all relative ${
              activeTab === tab.key
                ? 'text-primary font-bold'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            {tab.label}
            {activeTab === tab.key && (
              <motion.div
                className="absolute bottom-0 left-0 right-0 h-0.5"
                style={{ background: 'linear-gradient(90deg, hsl(56, 100%, 50%), hsl(180, 100%, 50%))' }}
                layoutId="tab-underline"
              />
            )}
          </button>
        ))}
      </div>
      {activeTab === 'analytics' ? <AnalyticsTab /> : <ReviewTab />}
    </div>
  );
}
