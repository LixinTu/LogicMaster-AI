import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { api, ReviewItem, isDemoMode } from '@/lib/api';
import { useAuthStore } from '@/store/useAuthStore';
import { Zap, Lightbulb } from 'lucide-react';

export default function Review() {
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
    return { color: 'hsl(345, 100%, 60%)', label: 'CRITICAL', textClass: 'text-destructive' };
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
      <div className="flex flex-col items-center justify-center h-[60vh] gap-6 yellow-atmosphere">
        <div className="loading-bulb-container">
          <Lightbulb size={64} className="loading-bulb-icon loading-bulb-fill" stroke="hsl(0, 0%, 0%)" strokeWidth={1.5} />
        </div>
        <p className="font-mono text-lg tracking-wider font-bold chromatic-text" style={{ color: 'hsl(0, 0%, 0%)' }}>
          Scanning memory banks...
        </p>
      </div>
    );
  }

  if (reviews.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-[60vh] gap-6 yellow-atmosphere">
        <div className="loading-bulb-container">
          <Lightbulb size={64} className="loading-bulb-icon loading-bulb-fill" stroke="hsl(0, 0%, 0%)" strokeWidth={1.5} />
        </div>
        <p className="font-mono text-lg tracking-wider font-bold chromatic-text" style={{ color: 'hsl(0, 0%, 0%)' }}>All systems nominal.</p>
        <p className="font-mono text-sm tracking-wider chromatic-text text-center max-w-md" style={{ color: 'hsl(0, 0%, 0%)' }}>
          No bugs detected in your memory banks. Come back tomorrow or practice new challenges →
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {isDemoMode() && (
        <div className="px-3 py-2 rounded-lg bg-primary/10 border border-primary/20 text-xs font-mono text-primary flex items-center gap-2">
          <span className="inline-block w-2 h-2 rounded-full bg-primary animate-pulse" />
          DEMO MODE — API Offline · Using sample data
        </div>
      )}

      <div className="flex items-center justify-between">
        <h1 className="font-mono text-xl font-bold tracking-wide uppercase text-primary-foreground chromatic-text">MEMORY MAINTENANCE</h1>
        <span className="px-4 py-1.5 rounded-full text-sm font-mono border border-border" style={{ backgroundColor: 'hsl(270, 100%, 8%)' }}>
          <span className="rainbow-logo-text">{dueCount} items need review</span>
        </span>
      </div>

      <div className="space-y-3">
        {reviews.map((item, i) => {
          const recall = getRecallInfo(item.recall_probability);
          return (
            <motion.div
              key={item.question_id}
              className="bg-card border border-border rounded-lg p-4"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.05 }}
            >
              <div className="flex items-start justify-between gap-4">
                <div className="flex-1 min-w-0 space-y-3">
                  <div className="flex flex-wrap items-center gap-2">
                    {item.question_type && (
                      <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase border ${getTypeBadgeClass(item.question_type)}`}>{item.question_type}</span>
                    )}
                    {item.difficulty && (
                      <span className={`px-2 py-0.5 rounded text-[10px] font-mono uppercase border ${getDifficultyBadgeClass(item.difficulty)}`}>{item.difficulty}</span>
                    )}
                    {item.skills?.map((skill, si) => (
                      <span key={si} className={`px-2 py-0.5 rounded text-[10px] font-mono border skill-badge-${si % 5}`}>{skill}</span>
                    ))}
                  </div>
                  {item.stimulus_preview && <p className="text-xs font-mono text-foreground/90 truncate">{item.stimulus_preview}</p>}
                  <div>
                    <div className="flex justify-between text-[10px] font-mono mb-1">
                      <span className={recall.textClass} style={{ fontWeight: 700 }}>{recall.label}</span>
                      <span className="text-foreground/90">{Math.round(item.recall_probability * 100)}%</span>
                    </div>
                    <div className="h-2 bg-muted rounded-full overflow-hidden">
                      <motion.div
                        className="h-full rounded-full"
                        style={{ backgroundColor: recall.color }}
                        initial={{ width: 0 }}
                        animate={{ width: `${item.recall_probability * 100}%` }}
                        transition={{ duration: 0.5, delay: i * 0.05 }}
                      />
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
