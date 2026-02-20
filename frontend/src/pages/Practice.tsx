import { useState, useEffect, useCallback } from 'react';
import { useLocation } from 'react-router-dom';
import { Lightbulb, Star } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useAppStore } from '@/store/useAppStore';
import { useAuthStore } from '@/store/useAuthStore';
import { api, isDemoMode } from '@/lib/api';
import { AnimatedNumber } from '@/components/AnimatedNumber';
import { Send, ArrowRight, Loader2 } from 'lucide-react';

const getTypeBadgeClass = (type: string) => {
  switch (type?.toLowerCase()) {
    case 'weaken': return 'bg-destructive/20 text-destructive border-destructive/30';
    case 'strengthen': return 'bg-success/20 text-success border-success/30';
    case 'assumption': return 'bg-primary/20 text-primary border-primary/30';
    case 'inference': return 'bg-secondary/20 text-secondary border-secondary/30';
    case 'flaw': return 'bg-warning/20 text-warning border-warning/30';
    case 'evaluate': return 'bg-primary/20 text-primary border-primary/30';
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

// ---- Explanation markdown renderer ----

function renderInlineBold(text: string) {
  const parts = text.split(/\*\*([^*]+)\*\*/);
  return parts.map((part, i) =>
    i % 2 === 1
      ? <strong key={i} style={{ color: 'hsl(56, 100%, 50%)', fontWeight: 700 }}>{part}</strong>
      : <span key={i}>{part}</span>
  );
}

function ExplanationMarkdown({ text }: { text: string }) {
  const lines = text.split('\n');
  const nodes: JSX.Element[] = [];
  let bullets: string[] = [];
  let k = 0;

  function flushBullets() {
    if (!bullets.length) return;
    nodes.push(
      <ul key={`ul${k++}`} style={{ listStyle: 'none', padding: 0, margin: '0 0 12px 0' }}>
        {bullets.map((item, bi) => (
          <li key={bi} style={{ display: 'flex', gap: '8px', alignItems: 'flex-start', marginBottom: '4px' }}>
            <span style={{ color: 'hsl(180, 100%, 50%)', flexShrink: 0, marginTop: '3px', fontSize: '10px' }}>▸</span>
            <span style={{ color: 'hsl(0, 0%, 85%)', fontSize: '14px', lineHeight: 1.6 }}>
              {renderInlineBold(item)}
            </span>
          </li>
        ))}
      </ul>
    );
    bullets = [];
  }

  for (const line of lines) {
    const t = line.trim();

    // Section header: ## or #
    if (/^#{1,3}\s/.test(t)) {
      flushBullets();
      nodes.push(
        <p key={`h${k++}`} style={{
          color: 'hsl(180, 100%, 60%)',
          fontSize: '10px',
          fontFamily: 'monospace',
          fontWeight: 700,
          letterSpacing: '0.12em',
          textTransform: 'uppercase',
          marginTop: nodes.length ? '18px' : 0,
          marginBottom: '6px',
        }}>
          {t.replace(/^#{1,3}\s+/, '')}
        </p>
      );
      continue;
    }

    // Bullet items
    if (/^[-*]\s/.test(t)) {
      bullets.push(t.slice(2));
      continue;
    }

    // Answer choice line: **A.** or **A)**
    const choiceMatch = t.match(/^\*\*([A-E])[.)]\*\*\s*(.*)/);
    if (choiceMatch) {
      flushBullets();
      nodes.push(
        <div key={`ch${k++}`} style={{ display: 'flex', gap: '10px', alignItems: 'flex-start', marginBottom: '6px' }}>
          <span style={{
            color: 'hsl(56, 100%, 50%)',
            fontFamily: 'monospace',
            fontWeight: 700,
            fontSize: '13px',
            flexShrink: 0,
            paddingTop: '1px',
            minWidth: '18px',
          }}>
            {choiceMatch[1]}.
          </span>
          <span style={{ color: 'hsl(0, 0%, 80%)', fontSize: '14px', lineHeight: 1.6 }}>
            {renderInlineBold(choiceMatch[2])}
          </span>
        </div>
      );
      continue;
    }

    // Empty line → flush bullets + paragraph spacing
    if (!t) {
      flushBullets();
      if (nodes.length > 0) {
        nodes.push(<div key={`sp${k++}`} style={{ height: '10px' }} />);
      }
      continue;
    }

    // Regular paragraph
    flushBullets();
    nodes.push(
      <p key={`p${k++}`} style={{ color: 'hsl(0, 0%, 85%)', fontSize: '14px', lineHeight: 1.6, marginBottom: '6px' }}>
        {renderInlineBold(t)}
      </p>
    );
  }

  flushBullets();
  return <div>{nodes}</div>;
}

// ---- End explanation renderer ----

const BLOOMS_STEPS = ['REM', 'UND', 'APP', 'ANL', 'EVL', 'CRT'];
const BLOOMS_FULL = ['Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', 'Create'];

const SKILL_BADGE_CLASSES = ['skill-badge-0', 'skill-badge-1', 'skill-badge-2', 'skill-badge-3', 'skill-badge-4'];

const TYPE_COLORS: Record<string, string> = {
  weaken: 'hsl(345, 100%, 60%)',
  strengthen: 'hsl(152, 100%, 50%)',
  assumption: 'hsl(56, 100%, 50%)',
  inference: 'hsl(180, 100%, 50%)',
  flaw: 'hsl(20, 100%, 60%)',
  evaluate: 'hsl(56, 100%, 50%)',
};

function getTypeColor(type: string) {
  return TYPE_COLORS[type?.toLowerCase()] || 'hsl(56, 100%, 50%)';
}

function getChoiceLetter(index: number) {
  return String.fromCharCode(65 + index);
}

function getSkillBadgeClass(index: number) {
  return SKILL_BADGE_CLASSES[index % SKILL_BADGE_CLASSES.length];
}

export default function Practice() {
  const store = useAppStore();
  const { getUserId } = useAuthStore();
  const location = useLocation();
  const [submitting, setSubmitting] = useState(false);
  const [flashColor, setFlashColor] = useState<string | null>(null);
  const [tutorInput, setTutorInput] = useState('');
  const [explanation, setExplanation] = useState<any>(null);
  const [isFavorited, setIsFavorited] = useState(false);
  const [tutoringComplete, setTutoringComplete] = useState(false);

  const fetchQuestion = useCallback(async (reviewQuestionId?: string) => {
    store.setPracticeState('loading');
    store.setSelectedChoice(null);
    let q;
    if (reviewQuestionId) {
      q = await api.getQuestionById(reviewQuestionId);
    } else {
      q = await api.nextQuestion(
        store.theta,
        store.currentQuestion?.question_id || '',
        store.questionsLog,
        'bandit'
      );
    }
    store.setCurrentQuestion(q);
    store.incrementSession();
    store.setPracticeState('answering');
  }, [store.theta, store.questionsLog]);

  useEffect(() => {
    const state = location.state as { reviewQuestionId?: string } | null;
    // Only fetch if we have a specific review question OR no question is loaded yet
    if (state?.reviewQuestionId || !store.currentQuestion) {
      fetchQuestion(state?.reviewQuestionId);
      if (state?.reviewQuestionId) {
        window.history.replaceState({}, '');
      }
    } else {
      // Already have a question loaded, just make sure we're in answering state
      if (store.practiceState === 'loading') {
        store.setPracticeState('answering');
      }
    }
  }, []);

  const finalizeAnswer = async (isCorrect: boolean) => {
    if (!store.currentQuestion) return;
    const userId = getUserId();
    const realTheta = await api.updateTheta(
      store.theta,
      store.currentQuestion.elo_difficulty,
      isCorrect
    );
    store.setTheta(realTheta.new_theta);
    store.setGmatScore(realTheta.gmat_score);
    store.recordAnswer(isCorrect);
    store.addQuestionLog({
      question_id: store.currentQuestion.question_id,
      is_correct: isCorrect,
      theta_at_time: store.theta,
    });
    api.banditUpdate(
      store.currentQuestion.question_id,
      isCorrect,
      store.currentQuestion.skills,
      store.theta,
      userId
    ).catch(console.error);

    // Auto-bookmark wrong answers
    if (!isCorrect) {
      api.bookmarkAdd(userId, store.currentQuestion.question_id, 'wrong').catch(console.error);
    }
  };

  const handleSubmit = async () => {
    if (!store.selectedChoice || !store.currentQuestion) return;
    setSubmitting(true);

    try {
      const selectedIdx = store.currentQuestion.choices.findIndex(c => c === store.selectedChoice);
      const correctLetter = store.currentQuestion.correct_answer;
      const correctIdx = correctLetter ? correctLetter.charCodeAt(0) - 65 : -1;
      const isCorrect = selectedIdx === correctIdx;

      if (isCorrect) {
        await finalizeAnswer(true);
        setFlashColor('hsl(152, 100%, 50%)');
        store.setPracticeState('correct');
      } else {
        const userChoice = store.selectedChoice; // Issue 3: capture before clearing
        setFlashColor('hsl(345, 100%, 60%)');
        store.setSelectedChoice(null);
        store.setPracticeState('retrying');
        startTutoring(userChoice);
      }

      setTimeout(() => setFlashColor(null), 300);
    } catch (err) {
      console.error('Submit error:', err);
    } finally {
      setSubmitting(false);
    }
  };

  const handleRetrySubmit = async () => {
    if (!store.selectedChoice || !store.currentQuestion) return;
    setSubmitting(true);

    try {
      const selectedIdx = store.currentQuestion.choices.findIndex(c => c === store.selectedChoice);
      const correctLetter = store.currentQuestion.correct_answer;
      const correctIdx = correctLetter ? correctLetter.charCodeAt(0) - 65 : -1;
      const isCorrect = selectedIdx === correctIdx;

      await finalizeAnswer(isCorrect);

      if (isCorrect) {
        setFlashColor('hsl(152, 100%, 50%)');
        store.setPracticeState('correct');
        // Show explanation even on correct retry so the user can learn from it
        if (!explanation) {
          try {
            const expResult = await api.generateExplanation(
              store.currentQuestion.question_id,
              store.currentQuestion,
              '',
              false
            );
            setExplanation(typeof expResult === 'string' ? expResult : expResult?.explanation || '');
          } catch {
            // silent fail
          }
        }
      } else {
        setFlashColor('hsl(345, 100%, 60%)');
        store.setPracticeState('wrong');
        // Always ensure explanation is available in the 'wrong' state
        if (!explanation) {
          try {
            const expResult = await api.generateExplanation(
              store.currentQuestion.question_id,
              store.currentQuestion,
              store.selectedChoice || '',
              false
            );
            setExplanation(typeof expResult === 'string' ? expResult : expResult?.explanation || '');
          } catch {
            // silent fail — card still renders without explanation
          }
        }
      }
      setTimeout(() => setFlashColor(null), 300);
    } catch (err) {
      console.error('Retry submit error:', err);
    } finally {
      setSubmitting(false);
    }
  };

  const startTutoring = async (userChoice: string | null) => {
    if (!store.currentQuestion || !userChoice) return;
    store.clearTutorMessages();
    setTutoringComplete(false);
    const userId = getUserId();
    const choiceLetter = userChoice.charAt(0); // Issue 4: send letter only, not full text

    // Issue 2: show panel immediately with loading placeholder
    store.addTutorMessage({ role: 'system', content: '⏳ Analyzing your logic...' });
    store.setHintCount(0); // Issue 1: hint_count starts at 0

    try {
      const res = await api.startRemediation(
        store.currentQuestion.question_id,
        store.currentQuestion,
        choiceLetter, // Issue 4
        store.currentQuestion.correct_answer,
        userId
      );
      store.setConversationId(res.conversation_id);
      // Issue 2: replace loading placeholder with actual diagnosis + hint
      store.clearTutorMessages();
      if (res.logic_gap) {
        store.addTutorMessage({ role: 'system', content: res.logic_gap });
      }
      if (res.first_hint) {
        store.addTutorMessage({ role: 'system', content: res.first_hint });
      }
      store.setHintCount(0); // Issue 1: always 0 after start-remediation
    } catch (err) {
      store.clearTutorMessages();
      store.addTutorMessage({ role: 'system', content: '⚠️ TUTOR CONNECTION INTERRUPTED — Analyzing your logic independently...' });
    }
  };

  const sendTutorMessage = async () => {
    if (!tutorInput.trim() || !store.conversationId || !store.currentQuestion || tutoringComplete) return;
    const msg = tutorInput.trim();
    setTutorInput('');
    store.addTutorMessage({ role: 'user', content: msg });
    store.setTutorLoading(true);

    try {
      const res = await api.continueTutor(store.conversationId, msg, store.currentQuestion.question_id);
      store.addTutorMessage({ role: 'system', content: res.reply });
      store.setBloomsLevel(res.blooms_level);
      store.setBloomsName(res.blooms_name);
      store.setHintCount(res.hint_count);

      if (!res.should_continue) {
        // Issue 7: do NOT call /conclude again (would generate a second LLM conclusion)
        // Issue 6: mark session complete, add completion message
        setTutoringComplete(true);
        store.addTutorMessage({
          role: 'system',
          content: '✅ TUTORING SESSION COMPLETE — Select the correct answer above to continue.',
        });
        // Issue 5: wire up RAG explanation
        try {
          const expResult = await api.generateExplanation(
            store.currentQuestion.question_id,
            store.currentQuestion,
            '',
            false
          );
          setExplanation(typeof expResult === 'string' ? expResult : expResult?.explanation || '');
        } catch {
          // explanation stays null if RAG fails
        }
      }
    } catch (err) {
      store.addTutorMessage({ role: 'system', content: '⚠️ Signal lost. Try again.' });
    } finally {
      store.setTutorLoading(false);
    }
  };

  const concludeTutoring = async () => {
    if (!store.conversationId || !store.currentQuestion) return;
    try {
      await api.concludeTutor(store.conversationId, store.currentQuestion.question_id);
    } catch (err) {
      console.error('Conclude error:', err);
    }
  };

  const handleNext = () => {
    setExplanation(null);
    setIsFavorited(false);
    setTutoringComplete(false);
    store.resetPracticeState();
    fetchQuestion();
  };

  const toggleFavorite = () => {
    if (!store.currentQuestion) return;
    const userId = getUserId();
    if (!isFavorited) {
      api.bookmarkAdd(userId, store.currentQuestion.question_id, 'favorite').catch(console.error);
    } else {
      api.bookmarkRemove(userId, store.currentQuestion.question_id).catch(console.error);
    }
    setIsFavorited(!isFavorited);
  };

  const q = store.currentQuestion;
  const typeColor = q ? getTypeColor(q.question_type) : 'hsl(56, 100%, 50%)';
  const correctIdx = q?.correct_answer ? q.correct_answer.charCodeAt(0) - 65 : -1;

  // LOADING STATE
  if (store.practiceState === 'loading') {
    return (
      <div className="flex flex-col items-center justify-center h-[60vh] gap-6 yellow-atmosphere">
        <div className="loading-bulb-container">
          <Lightbulb size={64} className="loading-bulb-icon loading-bulb-fill" stroke="hsl(0, 0%, 0%)" strokeWidth={1.5} />
        </div>
        <p className="font-mono text-lg tracking-wider font-bold chromatic-text" style={{ color: 'hsl(0, 0%, 0%)' }}>
          Transmitting next challenge...
        </p>
      </div>
    );
  }

  if (!q) return null;

  return (
    <div className="relative yellow-atmosphere">
      {isDemoMode() && (
        <div className="mb-4 px-3 py-2 rounded-lg bg-card border border-border text-xs font-mono text-muted-foreground flex items-center gap-2">
          <span className="inline-block w-2 h-2 rounded-full bg-primary animate-pulse" />
          DEMO MODE — API Offline · Using sample data
        </div>
      )}

      {/* Flash overlay */}
      <AnimatePresence>
        {flashColor && (
          <motion.div
            className="fixed inset-0 z-50 pointer-events-none"
            style={{ backgroundColor: flashColor }}
            initial={{ opacity: 0.25 }}
            animate={{ opacity: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
          />
        )}
      </AnimatePresence>

      {/* Top stats bar */}
      <motion.div
        className="flex items-center gap-3 mb-6 flex-wrap"
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <span className="font-mono text-sm text-primary px-2.5 py-0.5 rounded-full" style={{ backgroundColor: 'hsl(270, 100%, 8%)' }}>Q.{store.sessionQuestions}</span>
        <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase border ${getTypeBadgeClass(q.question_type)}`} style={{ backgroundColor: 'hsl(270, 100%, 8%)' }}>
          {q.question_type}
        </span>
        <span className={`px-2 py-0.5 rounded text-[10px] font-mono uppercase border ${getDifficultyBadgeClass(q.difficulty)}`} style={{ backgroundColor: 'hsl(270, 100%, 8%)' }}>
          {q.difficulty}
        </span>
        {q.skills.map((skill, i) => (
          <span key={skill} className={`px-2 py-0.5 rounded text-[10px] font-mono border ${getSkillBadgeClass(i)}`} style={{ backgroundColor: 'hsl(270, 100%, 8%)' }}>
            {skill}
          </span>
        ))}
      </motion.div>

      {/* Question Card */}
      <motion.div
        className="rounded-lg overflow-hidden mb-6 rainbow-left-border relative"
        style={{ backgroundColor: 'hsl(270, 100%, 8%)' }}
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.1 }}
      >
        <button
          onClick={toggleFavorite}
          className="absolute top-4 right-4 z-10 transition-colors"
        >
          <Star
            size={20}
            className={isFavorited ? 'text-primary fill-primary' : 'text-muted-foreground hover:text-primary'}
          />
        </button>
        <div className="p-6 pr-12">
          <p className="text-foreground/90 leading-relaxed mb-4 text-[15px]">{q.stimulus}</p>
          <div className="h-px bg-gradient-to-r from-primary/30 via-secondary/20 to-transparent mb-4" />
          <p className="text-foreground font-semibold text-base">{q.question}</p>
        </div>
      </motion.div>

      {/* Answer Choices */}
      <div className="space-y-2 mb-6">
        {q.choices.map((choice, idx) => {
          const letter = getChoiceLetter(idx);
          const isSelected = store.selectedChoice === choice;
          const canSelect = store.practiceState === 'answering' || store.practiceState === 'retrying';
          const isCorrectChoice = idx === correctIdx;
          const showResult = store.practiceState === 'correct' || store.practiceState === 'wrong';

          let borderColor = 'transparent';
          let bgStyle = 'hsl(270, 100%, 8%)';
          if (isSelected && canSelect) {
            borderColor = 'hsl(56, 100%, 50%)';
          }
          if (showResult && isCorrectChoice) {
            borderColor = 'hsl(152, 100%, 50%)';
          }
          if (showResult && isSelected && !isCorrectChoice) {
            borderColor = 'hsl(345, 100%, 60%)';
          }

          return (
            <motion.button
              key={idx}
              className={`w-full text-left flex items-start gap-3 p-4 rounded-lg border ${
                canSelect ? 'hover:border-primary/30 cursor-pointer' : 'cursor-default'
              }`}
              style={{ borderColor, borderWidth: '1px', borderLeftWidth: isSelected ? '3px' : '1px', backgroundColor: bgStyle }}
              onClick={() => canSelect && store.setSelectedChoice(choice)}
              initial={{ opacity: 0, x: -10 }}
              animate={{
                opacity: showResult && !isSelected && !isCorrectChoice ? 0.5 : 1,
                x: 0,
                scale: isSelected ? 1.01 : 1,
              }}
              transition={{ type: 'spring', stiffness: 300, damping: 25, delay: 0.05 * idx }}
              whileHover={canSelect ? { scale: 1.008, transition: { duration: 0.2 } } : undefined}
            >
              <motion.span
                className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-mono font-bold flex-shrink-0 mt-0.5 border"
                animate={{
                  borderColor: isSelected ? typeColor : 'hsl(var(--border))',
                  color: isSelected ? typeColor : 'hsl(var(--muted-foreground))',
                }}
                transition={{ duration: 0.25 }}
              >
                {letter}
              </motion.span>
              <motion.span
                className="leading-relaxed"
                initial={false}
                animate={{
                  color: isSelected ? 'hsl(var(--foreground))' : 'hsl(var(--foreground) / 0.85)',
                  fontWeight: isSelected ? 700 : 400,
                  fontSize: isSelected ? '1rem' : '0.875rem',
                }}
                transition={{ type: 'spring', stiffness: 200, damping: 20 }}
              >
                {choice.replace(/^[A-E]\.\s*/, '')}
              </motion.span>
            </motion.button>
          );
        })}
      </div>

      {/* Submit / Next buttons */}
      {(store.practiceState === 'answering' || store.practiceState === 'retrying') && (
        <motion.button
          className={`w-full py-3.5 rounded-lg font-heading text-sm font-bold uppercase tracking-widest transition-all ${
            store.selectedChoice
              ? 'rainbow-btn glow-hover'
              : 'bg-muted text-muted-foreground cursor-not-allowed'
          }`}
          onClick={store.practiceState === 'retrying' ? handleRetrySubmit : handleSubmit}
          disabled={!store.selectedChoice || submitting}
          whileTap={store.selectedChoice ? { scale: 0.99 } : undefined}
        >
          {submitting ? (
            <span className="flex items-center justify-center gap-2">
              <Loader2 size={16} className="animate-spin" /> Processing...
            </span>
          ) : store.practiceState === 'retrying' ? (
            'Try Again ⚡'
          ) : (
            'Submit Answer'
          )}
        </motion.button>
      )}

      {/* Correct feedback */}
      {store.practiceState === 'correct' && (
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
          <div className="glass-card rounded-lg p-4 mb-4" style={{ borderImage: 'linear-gradient(90deg, hsl(152, 100%, 50%, 0.5), hsl(56, 100%, 50%, 0.3)) 1' }}>
            <h3 className="font-heading text-lg tracking-wider rainbow-text chromatic-text">SYSTEM UPGRADED ⚡</h3>
            <div className="flex gap-6 mt-2 font-mono text-sm">
              <span className="text-muted-foreground">θ: <AnimatedNumber value={store.theta} decimals={2} className="text-secondary" /></span>
              <span className="text-muted-foreground">GMAT: <AnimatedNumber value={store.gmatScore} className="text-primary" /></span>
              <span className="text-muted-foreground">🔥 <AnimatedNumber value={store.streak} className="text-foreground" /></span>
            </div>
          </div>

          {explanation && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="glass-card rounded-lg p-5 mb-4"
              style={{ borderLeftWidth: '3px', borderLeftColor: 'hsl(180, 100%, 50%)' }}
            >
              <h4 className="font-mono text-xs font-bold tracking-widest mb-4" style={{ color: 'hsl(180, 100%, 60%)' }}>
                LOGIC PATCH NOTES
              </h4>
              <ExplanationMarkdown text={typeof explanation === 'string' ? explanation : JSON.stringify(explanation)} />
            </motion.div>
          )}

          <button
            className="w-full py-3 rounded-lg rainbow-btn font-heading text-sm uppercase tracking-widest glow-hover flex items-center justify-center gap-2"
            onClick={handleNext}
          >
            Next Challenge <ArrowRight size={16} />
          </button>
        </motion.div>
      )}

      {/* Retrying state */}
      {store.practiceState === 'retrying' && (
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
          <div className="glass-card rounded-lg p-4 mb-4" style={{ borderImage: 'linear-gradient(90deg, hsl(56, 100%, 50%, 0.5), hsl(180, 100%, 50%, 0.3)) 1' }}>
            <h3 className="font-heading text-lg tracking-wider rainbow-text chromatic-text">TRY AGAIN 🔄</h3>
            <p className="text-xs font-mono text-foreground/60 mt-1">You have one more chance. Use the hints below or select your answer directly.</p>
          </div>

          {(store.practiceState === 'retrying' || store.tutorMessages.length > 0) && (
            <div className="glass-card rounded-lg overflow-hidden mb-4" style={{ borderLeftWidth: '3px', borderLeftColor: 'hsl(180, 100%, 50%)' }}>
              <div className="p-4 border-b border-border flex items-center gap-2">
                <span className="font-heading text-secondary text-sm tracking-wider">LOGIC DEBUGGER</span>
                <span className="animate-[blink-cursor_1s_step-end_infinite] text-secondary">▌</span>
              </div>

              <div className="px-4 py-3 border-b border-border">
                <div className="flex gap-1">
                  {BLOOMS_FULL.map((step, i) => (
                    <div key={step} className="flex-1" title={step}>
                      <div
                        className={`h-1.5 rounded-full transition-all ${
                          i + 1 < store.bloomsLevel ? 'bg-success/50' :
                          i + 1 === store.bloomsLevel ? 'bg-primary animate-pulse' :
                          'bg-muted'
                        }`}
                      />
                      <span className="text-[8px] font-mono text-foreground/50 mt-0.5 block text-center">{step}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="px-4 py-2 border-b border-border flex items-center gap-2">
                <span className="text-[10px] font-mono text-foreground/50 uppercase">Hint Intensity:</span>
                <div className="flex gap-1">
                  {[1, 2, 3].map((i) => (
                    <div
                      key={i}
                      className={`w-2 h-2 rounded-full ${
                        i <= store.hintCount
                          ? i === 1 ? 'bg-primary' : i === 2 ? 'bg-warning' : 'bg-destructive'
                          : 'bg-muted'
                      }`}
                    />
                  ))}
                </div>
                <span className="text-[10px] font-mono text-foreground/50">{store.hintCount}/3</span>
              </div>

              <div className="p-4 space-y-3 max-h-[300px] overflow-y-auto">
                {store.tutorMessages.map((msg, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, x: msg.role === 'system' ? -10 : 10 }}
                    animate={{ opacity: 1, x: 0 }}
                    className={`text-sm font-mono ${msg.role === 'system' ? 'text-foreground/60' : 'text-foreground'}`}
                  >
                    <span className={msg.role === 'system' ? 'text-secondary' : 'text-primary'}>
                      {'> '}{msg.role === 'system' ? 'SYSTEM' : 'USER'}:
                    </span>{' '}
                    {msg.content}
                  </motion.div>
                ))}
                {store.tutorLoading && (
                  <div className="flex gap-1">
                    {[0, 1, 2].map((i) => (
                      <motion.div
                        key={i}
                        className="w-1.5 h-1.5 rounded-full bg-secondary"
                        animate={{ opacity: [0.3, 1, 0.3] }}
                        transition={{ duration: 0.8, delay: i * 0.2, repeat: Infinity }}
                      />
                    ))}
                  </div>
                )}
              </div>

              <div className="p-3 border-t border-border flex gap-2">
                <div className="flex-1 relative">
                  <span className="absolute left-3 top-1/2 -translate-y-1/2 text-primary font-mono text-xs">{'>'}</span>
                  <input
                    type="text"
                    value={tutorInput}
                    onChange={(e) => setTutorInput(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && sendTutorMessage()}
                    placeholder={tutoringComplete ? 'Session complete — select an answer above' : 'Type your reasoning...'}
                    disabled={tutoringComplete}
                    className="w-full bg-muted border border-primary/20 rounded px-7 py-2 text-sm font-mono text-foreground placeholder:text-muted-foreground focus:outline-none focus:border-primary/50 disabled:opacity-40 disabled:cursor-not-allowed"
                  />
                </div>
                <button
                  onClick={sendTutorMessage}
                  disabled={tutoringComplete}
                  className="px-3 py-2 bg-primary text-primary-foreground rounded glow-hover disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  <Send size={14} />
                </button>
              </div>
            </div>
          )}
        </motion.div>
      )}

      {/* Wrong state */}
      {store.practiceState === 'wrong' && (
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
          <div className="glass-card rounded-lg p-4 mb-4" style={{ borderImage: 'linear-gradient(90deg, hsl(345, 100%, 60%, 0.5), hsl(270, 80%, 60%, 0.3)) 1' }}>
            <h3 className="font-heading text-lg tracking-wider rainbow-text chromatic-text">GLITCH DETECTED ⚠️</h3>
            <p className="text-xs font-mono text-muted-foreground mt-1">Added to Wrong Book 📚</p>
          </div>

          {explanation && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="glass-card rounded-lg p-5 mb-4"
              style={{ borderLeftWidth: '3px', borderLeftColor: 'hsl(180, 100%, 50%)' }}
            >
              <h4 className="font-mono text-xs font-bold tracking-widest mb-4" style={{ color: 'hsl(180, 100%, 60%)' }}>
                LOGIC PATCH NOTES
              </h4>
              <ExplanationMarkdown text={typeof explanation === 'string' ? explanation : JSON.stringify(explanation)} />
            </motion.div>
          )}

          <button
            className="w-full py-3 rounded-lg rainbow-btn font-heading text-sm uppercase tracking-widest glow-hover flex items-center justify-center gap-2"
            onClick={handleNext}
          >
            Next Challenge <ArrowRight size={16} />
          </button>
        </motion.div>
      )}
    </div>
  );
}
