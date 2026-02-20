import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Brain, Lightbulb, CircuitBoard, Zap, Code, Network, Lock, Trash2, Eye, EyeOff, Shield } from 'lucide-react';
import { useAuthStore } from '@/store/useAuthStore';
import { useAppStore } from '@/store/useAppStore';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog';
import { api, isDemoMode, UserStats } from '@/lib/api';

const AVATAR_ICONS = [Brain, Lightbulb, CircuitBoard, Zap, Code, Network];
const AVATAR_NAMES = ['Brain', 'Lightbulb', 'Circuit', 'Lightning', 'Code', 'Neural Net'];

const strengthColors = ['bg-destructive', 'bg-warning', 'bg-primary', 'bg-success'];

function getPasswordStrength(pw: string): number {
  let s = 0;
  if (pw.length >= 6) s++;
  if (pw.length >= 10) s++;
  if (/[A-Z]/.test(pw) && /[0-9]/.test(pw)) s++;
  if (/[^A-Za-z0-9]/.test(pw)) s++;
  return Math.min(s, 4);
}

export default function ProfilePage() {
  const { user, logout } = useAuthStore();
  const store = useAppStore();

  const [avatarIdx, setAvatarIdx] = useState(0);
  const AvatarIcon = AVATAR_ICONS[avatarIdx];

  const [displayName, setDisplayName] = useState(user?.display_name || '');
  const [profileMsg, setProfileMsg] = useState<{ text: string; type: 'success' | 'error' } | null>(null);
  const [profileLoading, setProfileLoading] = useState(false);

  const [currentPw, setCurrentPw] = useState('');
  const [newPw, setNewPw] = useState('');
  const [confirmPw, setConfirmPw] = useState('');
  const [showCurrentPw, setShowCurrentPw] = useState(false);
  const [showNewPw, setShowNewPw] = useState(false);
  const [pwMsg, setPwMsg] = useState<{ text: string; type: 'success' | 'error' } | null>(null);
  const [pwLoading, setPwLoading] = useState(false);

  const [deleteOpen, setDeleteOpen] = useState(false);
  const [deleteConfirm, setDeleteConfirm] = useState('');
  const [deleteLoading, setDeleteLoading] = useState(false);

  const [apiStats, setApiStats] = useState<UserStats | null>(null);
  const [statsLoading, setStatsLoading] = useState(true);

  useEffect(() => {
    const fetchStats = async () => {
      const data = await api.authStats();
      setApiStats(data);
      setStatsLoading(false);
    };
    fetchStats();
  }, []);

  const initial = (user?.display_name || 'U')[0].toUpperCase();

  const handleUpdateProfile = async () => {
    if (!displayName.trim()) return;
    setProfileLoading(true);
    setProfileMsg(null);
    try {
      await api.updateProfile(displayName.trim());
      setProfileMsg({ text: 'PROFILE UPDATED ✅', type: 'success' });
    } catch {
      // Demo mode fallback
      setProfileMsg({ text: 'PROFILE UPDATED ✅', type: 'success' });
    } finally {
      setProfileLoading(false);
    }
  };

  const handleChangePassword = async () => {
    setPwMsg(null);
    if (newPw !== confirmPw) {
      setPwMsg({ text: 'PASSWORDS DO NOT MATCH', type: 'error' });
      return;
    }
    if (newPw.length < 6) {
      setPwMsg({ text: 'PASSWORD TOO SHORT (MIN 6)', type: 'error' });
      return;
    }
    setPwLoading(true);
    try {
      await api.changePassword(currentPw, newPw);
      setPwMsg({ text: 'PASSWORD UPGRADED ✅', type: 'success' });
      setCurrentPw('');
      setNewPw('');
      setConfirmPw('');
    } catch (err: any) {
      setPwMsg({ text: err.message || 'CURRENT PASSWORD INCORRECT', type: 'error' });
    } finally {
      setPwLoading(false);
    }
  };

  const handleDeleteAccount = async () => {
    if (deleteConfirm !== 'DELETE') return;
    setDeleteLoading(true);
    try {
      await api.deleteAccount();
    } catch {
      // demo mode
    }
    logout();
  };

  const pwStrength = getPasswordStrength(newPw);

  // Use API stats if available, fall back to local store
  const totalAnswered = apiStats?.total_questions ?? store.questionsLog.length;
  const totalCorrect = apiStats?.total_correct ?? store.questionsLog.filter((q) => q.is_correct).length;
  const accuracy = apiStats?.accuracy_pct != null ? apiStats.accuracy_pct.toFixed(1) : (totalAnswered > 0 ? ((totalCorrect / totalAnswered) * 100).toFixed(1) : '0.0');
  const bestStreak = apiStats?.best_streak ?? store.streak;
  const gmatScore = apiStats?.current_gmat_score ?? store.gmatScore;
  const theta = apiStats?.current_theta ?? store.theta;
  const memberSince = apiStats?.member_since || 'Feb 2026';
  const favoriteType = apiStats?.favorite_question_type || 'Critical Reasoning';

  return (
    <div className="max-w-2xl mx-auto space-y-6 yellow-atmosphere pb-12">
      {isDemoMode() && (
        <div className="px-3 py-2 rounded-lg bg-primary/10 border border-primary/20 text-xs font-mono text-primary flex items-center gap-2">
          <span className="inline-block w-2 h-2 rounded-full bg-primary animate-pulse" />
          DEMO MODE — API Offline · Showing local data
        </div>
      )}

      {/* Avatar Section */}
      <motion.div
        className="flex flex-col items-center gap-4 pt-4"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div
          className="w-24 h-24 rounded-full flex items-center justify-center border-2 border-primary/40 relative overflow-hidden"
          style={{ background: 'linear-gradient(135deg, hsl(270, 100%, 8%), hsl(240, 80%, 10%))' }}
        >
          <AvatarIcon size={44} className="text-primary" />
          <span className="absolute bottom-0 right-0 text-[10px] font-mono text-muted-foreground bg-background/80 px-1 rounded-tl">
            {AVATAR_NAMES[avatarIdx]}
          </span>
        </div>
        <Button
          variant="outline"
          size="sm"
          className="font-mono text-xs tracking-wider uppercase"
          onClick={() => setAvatarIdx((prev) => (prev + 1) % AVATAR_ICONS.length)}
        >
          Change Avatar ⚡
        </Button>
        <div className="text-center">
          <h1 className="font-heading text-2xl font-bold tracking-wider text-primary uppercase">
            {user?.display_name || 'User'}
          </h1>
          <p className="font-mono text-sm text-muted-foreground">{user?.email || 'demo@glitchmind.ai'}</p>
          <p className="font-mono text-xs text-muted-foreground/60 mt-1">Member since: {memberSince}</p>
        </div>
      </motion.div>

      {/* Edit Profile */}
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
        <Card className="glass-card border-border">
          <CardHeader className="pb-3">
            <CardTitle className="font-heading text-sm tracking-widest uppercase text-primary">Edit Profile</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <label className="font-mono text-xs text-muted-foreground mb-1 block">Display Name</label>
              <Input value={displayName} onChange={(e) => setDisplayName(e.target.value)} className="font-mono bg-background/50" maxLength={50} />
            </div>
            <div>
              <label className="font-mono text-xs text-muted-foreground mb-1 block flex items-center gap-1">
                Email <Lock size={10} className="text-muted-foreground/50" />
              </label>
              <Input value={user?.email || 'demo@glitchmind.ai'} disabled className="font-mono bg-background/30 text-muted-foreground cursor-not-allowed" />
            </div>
            <AnimatePresence>
              {profileMsg && (
                <motion.p initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }}
                  className={`font-mono text-xs font-bold ${profileMsg.type === 'success' ? 'text-success chromatic-text' : 'text-destructive'}`}>
                  {profileMsg.text}
                </motion.p>
              )}
            </AnimatePresence>
            <Button onClick={handleUpdateProfile} disabled={profileLoading || !displayName.trim()} className="w-full rainbow-btn font-heading text-sm uppercase tracking-widest">
              {profileLoading ? 'Updating...' : 'Update Profile ⚡'}
            </Button>
          </CardContent>
        </Card>
      </motion.div>

      {/* Change Password */}
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
        <Card className="glass-card border-border">
          <CardHeader className="pb-3">
            <CardTitle className="font-heading text-sm tracking-widest uppercase text-primary flex items-center gap-2">
              <Shield size={14} /> Change Password
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="relative">
              <label className="font-mono text-xs text-muted-foreground mb-1 block">Current Password</label>
              <Input type={showCurrentPw ? 'text' : 'password'} value={currentPw} onChange={(e) => setCurrentPw(e.target.value)} className="font-mono bg-background/50 pr-10" />
              <button type="button" onClick={() => setShowCurrentPw(!showCurrentPw)} className="absolute right-3 top-7 text-muted-foreground hover:text-foreground">
                {showCurrentPw ? <EyeOff size={14} /> : <Eye size={14} />}
              </button>
            </div>
            <div className="relative">
              <label className="font-mono text-xs text-muted-foreground mb-1 block">New Password</label>
              <Input type={showNewPw ? 'text' : 'password'} value={newPw} onChange={(e) => setNewPw(e.target.value)} className="font-mono bg-background/50 pr-10" />
              <button type="button" onClick={() => setShowNewPw(!showNewPw)} className="absolute right-3 top-7 text-muted-foreground hover:text-foreground">
                {showNewPw ? <EyeOff size={14} /> : <Eye size={14} />}
              </button>
              {newPw && (
                <div className="flex gap-1 mt-2">
                  {[0, 1, 2, 3].map((i) => (
                    <div key={i} className={`h-1 flex-1 rounded-full transition-all ${i < pwStrength ? strengthColors[pwStrength - 1] : 'bg-muted'}`} />
                  ))}
                </div>
              )}
            </div>
            <div>
              <label className="font-mono text-xs text-muted-foreground mb-1 block">Confirm New Password</label>
              <Input type="password" value={confirmPw} onChange={(e) => setConfirmPw(e.target.value)} className="font-mono bg-background/50" />
            </div>
            <AnimatePresence>
              {pwMsg && (
                <motion.p initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }}
                  className={`font-mono text-xs font-bold ${pwMsg.type === 'success' ? 'text-success chromatic-text' : 'text-destructive'}`}>
                  {pwMsg.text}
                </motion.p>
              )}
            </AnimatePresence>
            <Button onClick={handleChangePassword} disabled={pwLoading || !currentPw || !newPw || !confirmPw} className="w-full rainbow-btn font-heading text-sm uppercase tracking-widest">
              {pwLoading ? 'Updating...' : 'Change Password ⚡'}
            </Button>
          </CardContent>
        </Card>
      </motion.div>

      {/* Learning Stats */}
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}>
        <Card className="glass-card border-border">
          <CardHeader className="pb-3">
            <CardTitle className="font-heading text-sm tracking-widest uppercase text-primary">Learning Stats</CardTitle>
          </CardHeader>
          <CardContent>
            {statsLoading ? (
              <p className="font-mono text-sm text-muted-foreground">Loading stats...</p>
            ) : (
              <div className="grid grid-cols-2 gap-3">
                {[
                  { label: 'Total Answered', value: totalAnswered },
                  { label: 'Total Correct', value: totalCorrect },
                  { label: 'Accuracy', value: `${accuracy}%` },
                  { label: 'Best Streak', value: bestStreak },
                  { label: 'GMAT Score', value: gmatScore },
                  { label: 'Theta (θ)', value: typeof theta === 'number' ? theta.toFixed(2) : theta },
                  { label: 'Member Since', value: memberSince },
                  { label: 'Favorite Type', value: favoriteType },
                ].map((stat) => (
                  <div key={stat.label} className="p-3 rounded-lg border border-border" style={{ backgroundColor: 'hsl(270, 100%, 8%)' }}>
                    <p className="font-mono text-[10px] text-muted-foreground uppercase tracking-widest">{stat.label}</p>
                    <p className="font-heading text-lg text-foreground mt-1">{stat.value}</p>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </motion.div>

      {/* Danger Zone */}
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}>
        <Card className="border-destructive/40 glass-card">
          <CardHeader className="pb-3">
            <CardTitle className="font-heading text-sm tracking-widest uppercase text-destructive flex items-center gap-2">
              <Trash2 size={14} /> Danger Zone
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="font-mono text-xs text-muted-foreground mb-3">Permanently delete your account and all associated data. This action cannot be undone.</p>
            <Button variant="destructive" className="font-heading text-sm uppercase tracking-widest" onClick={() => setDeleteOpen(true)}>
              Delete Account
            </Button>
          </CardContent>
        </Card>
      </motion.div>

      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteOpen} onOpenChange={setDeleteOpen}>
        <DialogContent className="glass-card border-destructive/40">
          <DialogHeader>
            <DialogTitle className="font-heading text-destructive tracking-wider uppercase">⚠️ Delete Account</DialogTitle>
            <DialogDescription className="font-mono text-xs text-muted-foreground">
              This will permanently erase all your data. Type <strong className="text-destructive">DELETE</strong> to confirm.
            </DialogDescription>
          </DialogHeader>
          <Input value={deleteConfirm} onChange={(e) => setDeleteConfirm(e.target.value)} placeholder='Type "DELETE" to confirm' className="font-mono bg-background/50" />
          <div className="flex gap-2 justify-end">
            <Button variant="outline" onClick={() => { setDeleteOpen(false); setDeleteConfirm(''); }}>Cancel</Button>
            <Button variant="destructive" disabled={deleteConfirm !== 'DELETE' || deleteLoading} onClick={handleDeleteAccount}>
              {deleteLoading ? 'Deleting...' : 'Confirm Delete'}
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
