import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Lightbulb, Eye, EyeOff, Zap, ArrowRight } from 'lucide-react';
import { useAuthStore } from '@/store/useAuthStore';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';

export default function LoginPage() {
  const [isRegister, setIsRegister] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const login = useAuthStore((s) => s.login);
  const register = useAuthStore((s) => s.register);

  const passwordStrength = (() => {
    if (password.length === 0) return 0;
    let s = 0;
    if (password.length >= 6) s++;
    if (password.length >= 10) s++;
    if (/[A-Z]/.test(password) && /[a-z]/.test(password)) s++;
    if (/\d/.test(password)) s++;
    if (/[^A-Za-z0-9]/.test(password)) s++;
    return Math.min(s, 5);
  })();

  const strengthColors = [
    'hsl(0,60%,50%)',
    'hsl(15,70%,50%)',
    'hsl(40,80%,50%)',
    'hsl(56,90%,50%)',
    'hsl(120,60%,45%)',
  ];

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    if (isRegister && password !== confirmPassword) {
      setError('PASSWORDS DO NOT MATCH');
      return;
    }
    setLoading(true);
    try {
      if (isRegister) {
        await register(email, password, displayName);
      } else {
        await login(email, password);
      }
    } catch (err: any) {
      setError(err.message || 'AUTHENTICATION FAILED');
    } finally {
      setLoading(false);
    }
  };

  const switchMode = () => {
    setIsRegister(!isRegister);
    setError('');
  };

  return (
    <div
      className="min-h-screen flex items-center justify-center p-4 sidebar-rainbow-bg"
    >
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        className="w-full max-w-md glass-card p-8 rounded-xl border border-border"
      >
        {/* Logo */}
        <div className="flex flex-col items-center gap-2 mb-8">
          <Lightbulb size={40} className="text-primary" style={{ filter: 'drop-shadow(0 0 12px hsl(56,100%,50%))' }} />
          <div className="font-heading text-lg font-bold tracking-wider rainbow-logo-text uppercase">GlitchMind</div>
        </div>

        {/* Heading */}
        <h1 className="text-center font-heading text-2xl font-bold tracking-wider mb-6 rainbow-text">
          {isRegister ? 'INITIALIZE NEW MIND' : 'ACCESS YOUR MIND'}
        </h1>

        {/* Error */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="mb-4 p-3 rounded-md border border-destructive/50 text-center"
              style={{ backgroundColor: 'hsla(0,60%,50%,0.1)' }}
            >
              <span className="font-mono text-sm text-destructive chromatic-text">{error}</span>
            </motion.div>
          )}
        </AnimatePresence>

        <form onSubmit={handleSubmit} className="space-y-4">
          {isRegister && (
            <div>
              <label className="block font-mono text-xs text-muted-foreground mb-1.5 uppercase tracking-wider">Display Name</label>
              <Input
                value={displayName}
                onChange={(e) => setDisplayName(e.target.value)}
                placeholder="Enter display name"
                className="bg-background/50 border-border focus:border-primary transition-colors"
                required
              />
            </div>
          )}

          <div>
            <label className="block font-mono text-xs text-muted-foreground mb-1.5 uppercase tracking-wider">Email</label>
            <Input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="Enter your email"
              className="bg-background/50 border-border focus:border-primary transition-colors"
              required
            />
          </div>

          <div>
            <label className="block font-mono text-xs text-muted-foreground mb-1.5 uppercase tracking-wider">Password</label>
            <div className="relative">
              <Input
                type={showPassword ? 'text' : 'password'}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Enter password"
                className="bg-background/50 border-border focus:border-primary transition-colors pr-10"
                required
                minLength={6}
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
              >
                {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
            {isRegister && password.length > 0 && (
              <div className="flex gap-1 mt-2">
                {Array.from({ length: 5 }).map((_, i) => (
                  <div
                    key={i}
                    className="h-1 flex-1 rounded-full transition-colors duration-300"
                    style={{
                      backgroundColor: i < passwordStrength ? strengthColors[passwordStrength - 1] : 'hsl(var(--muted))',
                    }}
                  />
                ))}
              </div>
            )}
          </div>

          {isRegister && (
            <div>
              <label className="block font-mono text-xs text-muted-foreground mb-1.5 uppercase tracking-wider">Confirm Password</label>
              <Input
                type="password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                placeholder="Confirm password"
                className="bg-background/50 border-border focus:border-primary transition-colors"
                required
              />
            </div>
          )}

          <Button
            type="submit"
            disabled={loading}
            className="w-full font-mono font-bold tracking-wider text-base py-5 bg-primary text-primary-foreground hover:brightness-110 transition-all"
          >
            {loading ? (
              <div className="rainbow-spinner w-5 h-5" />
            ) : (
              <>
                {isRegister ? 'CREATE ACCOUNT' : 'LOGIN'} <Zap size={16} className="ml-1" />
              </>
            )}
          </Button>
        </form>

        <div className="mt-6 text-center">
          <button onClick={switchMode} className="font-mono text-xs text-muted-foreground hover:text-foreground transition-colors group">
            {isRegister ? 'Already have access? ' : 'No account? '}
            <span className="text-primary group-hover:underline">
              {isRegister ? 'LOGIN' : 'CREATE ONE'} <ArrowRight size={12} className="inline" />
            </span>
          </button>
        </div>
      </motion.div>
    </div>
  );
}
