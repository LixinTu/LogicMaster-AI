import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Bug, Lightbulb, X, Zap, Filter } from 'lucide-react';
import { useAuthStore } from '@/store/useAuthStore';
import { api, isDemoMode, BookmarkItem, BookmarkStats } from '@/lib/api';

const TYPE_BORDER_COLORS: Record<string, string> = {
  Weaken: 'hsl(345, 100%, 60%)',
  Strengthen: 'hsl(152, 100%, 50%)',
  Assumption: 'hsl(210, 80%, 55%)',
  Inference: 'hsl(270, 80%, 60%)',
  Flaw: 'hsl(20, 100%, 60%)',
};

const getTypeBadgeClass = (type: string) => {
  switch (type?.toLowerCase()) {
    case 'weaken': return 'bg-destructive/20 text-destructive border-destructive/30';
    case 'strengthen': return 'bg-success/20 text-success border-success/30';
    case 'assumption': return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
    case 'inference': return 'bg-purple-500/20 text-purple-400 border-purple-500/30';
    case 'flaw': return 'bg-warning/20 text-warning border-warning/30';
    default: return 'bg-muted text-muted-foreground border-border';
  }
};

const getDifficultyBadgeClass = (diff: string) => {
  switch (diff?.toLowerCase()) {
    case 'hard': return 'bg-destructive/10 text-destructive/80 border-destructive/20';
    case 'medium': return 'bg-warning/10 text-warning/80 border-warning/20';
    case 'easy': return 'bg-success/10 text-success/80 border-success/20';
    default: return 'bg-muted text-muted-foreground border-border';
  }
};

const SKILL_OPTIONS = ['All Skills', 'Causal Reasoning', 'Assumption ID', 'Alternative Explanations', 'Evidence Evaluation', 'Logical Structure'];
const TYPE_OPTIONS = ['All Types', 'Weaken', 'Strengthen', 'Assumption', 'Inference', 'Flaw'];
const BOOKMARK_OPTIONS = ['All', '🔴 Wrong Answers', '⭐ Favorites'];

export default function WrongBook() {
  const navigate = useNavigate();
  const { getUserId } = useAuthStore();
  const [bookmarks, setBookmarks] = useState<BookmarkItem[]>([]);
  const [stats, setStats] = useState<BookmarkStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [filterType, setFilterType] = useState('All');
  const [filterSkill, setFilterSkill] = useState('All Skills');
  const [filterQType, setFilterQType] = useState('All Types');

  useEffect(() => {
    const fetchData = async () => {
      const userId = getUserId();
      const [bk, st] = await Promise.all([
        api.bookmarksList(userId),
        api.bookmarksStats(userId),
      ]);
      setBookmarks(bk);
      setStats(st);
      setLoading(false);
    };
    fetchData();
  }, []);

  const filtered = bookmarks.filter((b) => {
    if (filterType === '🔴 Wrong Answers' && b.bookmark_type !== 'wrong') return false;
    if (filterType === '⭐ Favorites' && b.bookmark_type !== 'favorite') return false;
    if (filterSkill !== 'All Skills' && !b.skills.some(s => s.toLowerCase().includes(filterSkill.toLowerCase()))) return false;
    if (filterQType !== 'All Types' && b.question_type !== filterQType) return false;
    return true;
  });

  const handleRemove = async (id: string) => {
    const userId = getUserId();
    setBookmarks(prev => prev.filter(b => b.question_id !== id));
    api.bookmarkRemove(userId, id).catch(console.error);
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-[60vh] gap-6 yellow-atmosphere">
        <div className="loading-bulb-container">
          <Lightbulb size={64} className="loading-bulb-icon loading-bulb-fill" stroke="hsl(0, 0%, 0%)" strokeWidth={1.5} />
        </div>
        <p className="font-mono text-lg tracking-wider font-bold chromatic-text" style={{ color: 'hsl(0, 0%, 0%)' }}>Loading wrong book...</p>
      </div>
    );
  }

  const selectClass = "bg-muted border border-border rounded px-3 py-2 text-xs font-mono text-foreground focus:outline-none focus:border-primary/50 appearance-none cursor-pointer";

  return (
    <div className="space-y-6 yellow-atmosphere">
      {isDemoMode() && (
        <div className="px-3 py-2 rounded-lg bg-primary/10 border border-primary/20 text-xs font-mono text-primary flex items-center gap-2">
          <span className="inline-block w-2 h-2 rounded-full bg-primary animate-pulse" />
          DEMO MODE — API Offline · Using sample data
        </div>
      )}

      {/* Stats overview */}
      <div className="grid grid-cols-3 gap-4">
        <motion.div className="bg-card border border-border rounded-lg p-4 text-center" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
          <span className="text-[10px] font-mono text-muted-foreground uppercase tracking-widest">Total Wrong</span>
          <div className="text-3xl font-heading font-bold text-destructive mt-1">{stats?.total_wrong ?? 0}</div>
        </motion.div>
        <motion.div className="bg-card border border-border rounded-lg p-4 text-center" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05 }}>
          <span className="text-[10px] font-mono text-muted-foreground uppercase tracking-widest">Most Missed Skill</span>
          <div className="text-sm font-heading font-bold text-foreground mt-2">{stats?.by_skill?.[0]?.skill_name || 'N/A'} {stats?.by_skill?.[0]?.count ? <span className="text-muted-foreground">({stats.by_skill[0].count})</span> : null}</div>
        </motion.div>
        <motion.div className="bg-card border border-border rounded-lg p-4 text-center" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
          <span className="text-[10px] font-mono text-muted-foreground uppercase tracking-widest">Most Missed Type</span>
          <div className="text-sm font-heading font-bold text-foreground mt-2">{stats?.by_type?.[0]?.question_type || 'N/A'} {stats?.by_type?.[0]?.count ? <span className="text-muted-foreground">({stats.by_type[0].count})</span> : null}</div>
        </motion.div>
      </div>

      {/* Filters */}
      <motion.div className="flex gap-3 flex-wrap items-center" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.15 }}>
        <Filter size={14} className="text-muted-foreground" />
        <select value={filterType} onChange={(e) => setFilterType(e.target.value)} className={selectClass}>
          {BOOKMARK_OPTIONS.map(o => <option key={o} value={o}>{o}</option>)}
        </select>
        <select value={filterSkill} onChange={(e) => setFilterSkill(e.target.value)} className={selectClass}>
          {SKILL_OPTIONS.map(o => <option key={o} value={o}>{o}</option>)}
        </select>
        <select value={filterQType} onChange={(e) => setFilterQType(e.target.value)} className={selectClass}>
          {TYPE_OPTIONS.map(o => <option key={o} value={o}>{o}</option>)}
        </select>
      </motion.div>

      {/* Question list */}
      {filtered.length === 0 ? (
        <motion.div className="flex flex-col items-center justify-center h-[40vh] gap-4" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          <div className="loading-bulb-container">
            <Lightbulb size={48} className="loading-bulb-icon loading-bulb-fill" stroke="hsl(0, 0%, 0%)" strokeWidth={1.5} />
          </div>
          <p className="font-mono text-sm text-muted-foreground">No bugs in your collection yet.</p>
          <p className="font-mono text-xs text-muted-foreground/70">
            Wrong answers are automatically saved here.{' '}
            <button onClick={() => navigate('/practice')} className="text-primary hover:underline">Go practice! →</button>
          </p>
        </motion.div>
      ) : (
        <AnimatePresence>
          <div className="space-y-3">
            {filtered.map((bk, i) => (
              <motion.div
                key={bk.question_id}
                className="bg-card border border-border rounded-lg p-4 relative overflow-hidden"
                style={{ borderLeftWidth: '4px', borderLeftColor: TYPE_BORDER_COLORS[bk.question_type] || 'hsl(260, 60%, 30%)' }}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 100, height: 0 }}
                transition={{ delay: i * 0.03 }}
              >
                <div className="flex flex-wrap items-center gap-2 mb-2">
                  <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase border ${getTypeBadgeClass(bk.question_type)}`}>{bk.question_type}</span>
                  <span className={`px-2 py-0.5 rounded text-[10px] font-mono uppercase border ${getDifficultyBadgeClass(bk.difficulty)}`}>{bk.difficulty}</span>
                  {bk.skills.map((skill, si) => (
                    <span key={si} className={`px-2 py-0.5 rounded text-[10px] font-mono border skill-badge-${si % 5}`}>{skill}</span>
                  ))}
                </div>
                <p className="text-xs font-mono text-foreground/80 mb-3 line-clamp-2">{bk.stimulus_preview}</p>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <span className={`text-[10px] font-mono font-bold uppercase ${bk.bookmark_type === 'wrong' ? 'text-destructive' : 'text-primary'}`}>
                      {bk.bookmark_type === 'wrong' ? '🔴 WRONG' : '⭐ FAVORITE'}
                    </span>
                    <span className="text-[10px] font-mono text-muted-foreground">
                      {new Date(bk.created_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
                    </span>
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={() => navigate('/practice', { state: { reviewQuestionId: bk.question_id } })}
                      className="px-3 py-1.5 rounded text-xs font-mono font-bold uppercase rainbow-btn glow-hover flex items-center gap-1"
                    >
                      <Zap size={10} /> Redo
                    </button>
                    <button
                      onClick={() => handleRemove(bk.question_id)}
                      className="px-3 py-1.5 rounded text-xs font-mono text-muted-foreground hover:text-destructive border border-border hover:border-destructive/30 transition-colors flex items-center gap-1"
                    >
                      <X size={10} /> Remove
                    </button>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </AnimatePresence>
      )}
    </div>
  );
}
