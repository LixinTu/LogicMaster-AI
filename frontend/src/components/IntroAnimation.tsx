import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Lightbulb } from 'lucide-react';

const TOTAL_DURATION = 3500;
const GLITCH_COLORS = ['hsl(0,100%,50%)', 'hsl(180,100%,50%)', 'hsl(300,100%,50%)'];
const LOGO_TEXT = 'GlitchMind';
const TAGLINE = 'Fix the glitch. Master the logic.';

export function IntroAnimation({ onComplete }: { onComplete: () => void }) {
  const [phase, setPhase] = useState(0);
  const [typedChars, setTypedChars] = useState(0);
  const [showTagline, setShowTagline] = useState(false);
  const [exiting, setExiting] = useState(false);

  const skip = useCallback(() => {
    sessionStorage.setItem('intro_played', 'true');
    onComplete();
  }, [onComplete]);

  useEffect(() => {
    // Phase transitions
    const timers = [
      setTimeout(() => setPhase(1), 500),    // glitch burst
      setTimeout(() => setPhase(2), 1500),   // logo reveal
      setTimeout(() => setPhase(3), 2500),   // text reveal
      setTimeout(() => setExiting(true), 3200),
      setTimeout(() => {
        sessionStorage.setItem('intro_played', 'true');
        onComplete();
      }, TOTAL_DURATION),
    ];
    return () => timers.forEach(clearTimeout);
  }, [onComplete]);

  // Typewriter effect for phase 3
  useEffect(() => {
    if (phase < 3) return;
    if (typedChars >= LOGO_TEXT.length) {
      const t = setTimeout(() => setShowTagline(true), 200);
      return () => clearTimeout(t);
    }
    const t = setTimeout(() => setTypedChars((c) => c + 1), 80);
    return () => clearTimeout(t);
  }, [phase, typedChars]);

  return (
    <motion.div
      className="fixed inset-0 z-[9999] flex items-center justify-center overflow-hidden"
      style={{ backgroundColor: 'hsl(0,0%,0%)' }}
      animate={exiting ? { scale: 0.9, opacity: 0 } : {}}
      transition={{ duration: 0.3, ease: 'easeOut' }}
    >
      {/* Phase 1 — Black Void (handled by default black bg) */}

      {/* Phase 2 — Glitch Burst */}
      <AnimatePresence>
        {phase === 1 && (
          <>
            {GLITCH_COLORS.map((color, i) => (
              <motion.div
                key={color}
                className="absolute inset-0"
                initial={{ opacity: 0 }}
                animate={{
                  opacity: [0, 0.8, 0],
                  backgroundColor: [color, 'hsl(0,0%,0%)'],
                }}
                transition={{
                  duration: 0.15,
                  delay: i * 0.12,
                  ease: 'linear',
                }}
              />
            ))}
            {/* Scan lines racing */}
            {[0, 1, 2, 3, 4].map((i) => (
              <motion.div
                key={`scan-${i}`}
                className="absolute left-0 right-0 h-[2px]"
                style={{ backgroundColor: 'hsla(0,0%,100%,0.3)', top: `${15 + i * 18}%` }}
                initial={{ x: '-100%' }}
                animate={{ x: '100%' }}
                transition={{ duration: 0.3, delay: i * 0.08, ease: 'linear' }}
              />
            ))}
            {/* Center power-on line */}
            <motion.div
              className="absolute top-1/2 left-1/2 h-[2px] -translate-y-1/2"
              style={{ backgroundColor: 'hsl(0,0%,100%)' }}
              initial={{ width: 0, x: '-50%' }}
              animate={{ width: '100vw', x: '-50vw' }}
              transition={{ duration: 0.4, delay: 0.4, ease: 'easeOut' }}
            />
          </>
        )}
      </AnimatePresence>

      {/* Phase 3 — Logo Reveal */}
      {phase >= 2 && (
        <div className="flex flex-col items-center gap-6 relative z-10">
          {/* Lightbulb icon */}
          <motion.div
            className="relative"
            initial={{ filter: 'blur(12px) saturate(3)', opacity: 0 }}
            animate={{
              filter: 'blur(0px) saturate(1)',
              opacity: [0, 0.3, 1, 0.4, 1],
            }}
            transition={{ duration: 1, ease: 'easeOut' }}
          >
            {/* Shockwave glow */}
            <motion.div
              className="absolute inset-0 rounded-full"
              style={{
                background: 'radial-gradient(circle, hsla(56,100%,50%,0.6) 0%, transparent 70%)',
              }}
              initial={{ scale: 1, opacity: 0 }}
              animate={{ scale: [1, 8], opacity: [0.8, 0] }}
              transition={{ duration: 0.6, delay: 0.8, ease: 'easeOut' }}
            />
            <Lightbulb
              size={72}
              className="text-primary"
              style={{ filter: 'drop-shadow(0 0 20px hsl(56,100%,50%))' }}
            />
          </motion.div>

          {/* Phase 4 — Typewriter text */}
          {phase >= 3 && (
            <div className="flex flex-col items-center gap-3">
              <div className="font-mono text-3xl font-bold tracking-widest uppercase">
                {LOGO_TEXT.slice(0, typedChars).split('').map((char, i) => (
                  <motion.span
                    key={i}
                    className="inline-block rainbow-logo-text"
                    initial={{
                      textShadow: '2px 0 hsl(0,100%,50%), -2px 0 hsl(180,100%,50%)',
                    }}
                    animate={{
                      textShadow: '0 0 transparent',
                    }}
                    transition={{ duration: 0.1, delay: 0.05 }}
                  >
                    {char}
                  </motion.span>
                ))}
                <motion.span
                  className="inline-block w-[2px] h-7 ml-1 align-middle"
                  style={{ backgroundColor: 'hsl(var(--primary))' }}
                  animate={{ opacity: [1, 0] }}
                  transition={{ duration: 0.5, repeat: Infinity }}
                />
              </div>
              <AnimatePresence>
                {showTagline && (
                  <motion.p
                    className="font-mono text-sm text-muted-foreground tracking-wider"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.4 }}
                  >
                    {TAGLINE}
                  </motion.p>
                )}
              </AnimatePresence>
            </div>
          )}
        </div>
      )}

      {/* Skip button */}
      <button
        onClick={skip}
        className="absolute bottom-6 right-8 font-mono text-xs text-foreground/30 hover:text-foreground/60 transition-colors"
      >
        Skip
      </button>
    </motion.div>
  );
}
