package be.kuleuven.pylos.player.student;

import be.kuleuven.pylos.game.PylosGameIF;
import be.kuleuven.pylos.game.PylosGameSimulator;
import be.kuleuven.pylos.game.PylosGameState;
import be.kuleuven.pylos.game.PylosBoard;
import be.kuleuven.pylos.game.PylosLocation;
import be.kuleuven.pylos.game.PylosSquare;
import be.kuleuven.pylos.game.PylosSphere;
import be.kuleuven.pylos.game.PylosPlayerColor;
import be.kuleuven.pylos.player.PylosPlayer;
import java.util.Random;
import java.util.List;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Objects;


public class StudentPlayerThomas extends PylosPlayer {
    ActionGenerator actionGenerator = new ActionGenerator();
    @Override
    public void doMove(PylosGameIF game, PylosBoard board) {
        Random random = this.getRandom();
        Move move = actionGenerator.generateMove(board, this.PLAYER_COLOR, random);
        game.moveSphere(move.sphere, move.location);
    }

    @Override
    public void doRemove(PylosGameIF game, PylosBoard board) {
        Random random = this.getRandom();
        Remove remove = actionGenerator.generateRemove(board, this.PLAYER_COLOR, random);
        game.removeSphere(Objects.requireNonNull(remove.sphere, "Remove mag geen null zijn in REMOVE_FIRST"));
    }

    @Override
    public void doRemoveOrPass(PylosGameIF game, PylosBoard board) {
        Random random = this.getRandom();
        Remove remove = actionGenerator.generateRemovePass(board, this.PLAYER_COLOR, random);
        if (remove.sphere != null) {
            game.removeSphere(remove.sphere);
        } else {
            game.pass();
        }
    }
}

/* ========================== Actietypes ========================== */

abstract class Action { PylosSphere sphere; }

class Move extends Action {
    PylosLocation location;
    Move(PylosSphere sphere, PylosLocation location) { this.sphere = sphere; this.location = location; }
}

class Remove extends Action {
    Remove(PylosSphere sphere) { this.sphere = sphere; }
}

/* ========================== Generator met veralgemeende minimax ========================== */

class ActionGenerator {

    private static final int MAX_DEPTH = 60;
    private PylosPlayerColor botColor;
    private final java.util.Map<Long, Double> cache = new java.util.LinkedHashMap<>();
    private final java.util.Map<Long, Integer> evalCache = new java.util.HashMap<>();

    // --- Rollout ---ll
    private static final int SIMULATIONS = 15;
    private static final int ROLLOUT_MAX_PLIES = 10;
    private static final double ROLLOUT_EPSILON = 0.005;
    private static final int ROLLOUT_TOP_K = 3;

    // dit is voor de random number generator voor de monte carlo rollouts
    private final java.util.SplittableRandom fastRnd = new java.util.SplittableRandom(0xC0FFEE);
    private final java.util.ArrayDeque<Step> rolloutStack = new java.util.ArrayDeque<>(64);
    private final java.util.ArrayList<Action> actionBuf = new java.util.ArrayList<>(64);

    // Track current search limit for iterative deepening
    private int currentMaxDepth = 1;

    // Weights tuned door verschillende testen met verschillende waarden en dan regressie toegepast op de resultaten
    // de regressie vergelijking is dan gebruikt om de weights te bepalen, tot 100 procent, en dan telkens uitgetest wat de beste waarden zijn
    private static final int WEIGHT_HEIGHT =1;
    private static final int WEIGHT_SQUARE = 15;
    private static final int WEIGHT_ALMOST_SQUARE = 20;
    private static final int WEIGHT_CENTER = 1;
    private static final int WEIGHT_RESERVE = 50;
    private static final int WEIGHT_SUPPORT = 1;
    private static final int WEIGHT_THREAT = 10;

    //Zobrist hashing voor transpositie tabel
    private static final long[] ZOBRIST_TURN = new long[2];
    private static final long[][] ZOBRIST = new long[30][2];
    private static final long[][] ZOBRIST_RESERVE_COUNT = new long[2][16];
    static {
        Random rng = new Random(123456);
        for (int i = 0; i < 30; i++) {
            for (int j = 0; j < 2; j++) {
                ZOBRIST[i][j] = rng.nextLong();
            }
        }
        for (int p = 0; p < 2; p++) {
            for (int n = 0; n < 16; n++) {
                ZOBRIST_RESERVE_COUNT[p][n] = rng.nextLong();
            }
        }
        ZOBRIST_TURN[0] = rng.nextLong();
        ZOBRIST_TURN[1] = rng.nextLong();
    }

    private double previousBestScore = Double.NEGATIVE_INFINITY;
    private double lastComputedScore = Double.NEGATIVE_INFINITY;
    private static final double MARGIN = 34.0;

    Move generateMove(PylosBoard board, PylosPlayerColor color, Random random) {
        botColor = color;
        cache.clear();
        evalCache.clear();
        return minimaxEntry(board, PylosGameState.MOVE, color, Move.class, random);
    }

    Remove generateRemove(PylosBoard board, PylosPlayerColor color, Random random) {
        botColor = color;
        cache.clear();
        evalCache.clear();
        return minimaxEntry(board, PylosGameState.REMOVE_FIRST, color, Remove.class, random);
    }

    Remove generateRemovePass(PylosBoard board, PylosPlayerColor color, Random random) {
        botColor = color;
        cache.clear();
        evalCache.clear();
        return minimaxEntry(board, PylosGameState.REMOVE_SECOND, color, Remove.class, random);
    }

    /* ========================== Veralgemeende root minimax ========================== */

    private <T extends Action> T minimaxEntry(PylosBoard board, PylosGameState startState, PylosPlayerColor color, Class<T> expectedType, Random random) {
        PylosGameSimulator sim = new PylosGameSimulator(startState, color, board);
        List<Action> rootActions = generateActionsForState(board, sim.getState(), color);
        List<Action> bestActions = new ArrayList<>();

        if (rootActions.isEmpty()) {
            if (expectedType == Move.class) return expectedType.cast(new Move(board.getReserve(color), firstUsable(board)));
            if (expectedType == Remove.class) return expectedType.cast(new Remove(null));
            return null;
        }

        // Sorteer acties op heuristic voor betere alpha-beta pruning
        rootActions.sort((a, b) -> Double.compare(actionHeuristic(b), actionHeuristic(a)));

        // Aspiration window
        double alpha, beta;
        if (Double.isFinite(previousBestScore)) {
            alpha = previousBestScore - MARGIN;
            beta  = previousBestScore + MARGIN;
        } else {
            alpha = Double.NEGATIVE_INFINITY;
            beta  = Double.POSITIVE_INFINITY;
        }

        double bestScore = Double.NEGATIVE_INFINITY;
        for (Action a : rootActions) {
            if (!expectedType.isInstance(a)) continue;

            double score;
            PylosGameState prevState = sim.getState();

            if (a instanceof Move m) {
                boolean wasReserve = m.sphere.isReserve();
                PylosLocation oldLoc = wasReserve ? null : m.sphere.getLocation();

                sim.moveSphere(m.sphere, m.location);
                score = minimax(board, sim, 1, nextTurnAfter(prevState, sim.getState(), color), alpha, beta);

                if (wasReserve) sim.undoAddSphere(m.sphere, prevState, color);
                else sim.undoMoveSphere(m.sphere, oldLoc, prevState, color);

            } else if (a instanceof Remove r) {
                if (prevState == PylosGameState.REMOVE_SECOND && r.sphere == null) {
                    sim.pass();
                    score = minimax(board, sim, 1, nextTurnAfter(prevState, sim.getState(), color), alpha, beta);
                    sim.undoPass(prevState, color);
                } else {
                    PylosLocation oldLoc = r.sphere.getLocation();
                    sim.removeSphere(r.sphere);
                    score = minimax(board, sim, 1, nextTurnAfter(prevState, sim.getState(), color), alpha, beta);

                    if (prevState == PylosGameState.REMOVE_FIRST) {
                        sim.undoRemoveFirstSphere(r.sphere, oldLoc, prevState, color);
                    } else if (prevState == PylosGameState.REMOVE_SECOND) {
                        sim.undoRemoveSecondSphere(r.sphere, oldLoc, prevState, color);
                    }
                }
            } else continue;

            if (score > bestScore) {
                bestScore = score;
                alpha = bestScore;
                bestActions.clear();
                bestActions.add(a);
            }
            else if (score == bestScore) {
                bestActions.add(a);
            }

            if (alpha >= beta) break;
        }


        if (bestActions.size() > 10) {
            Collections.shuffle(bestActions, random);
        }
        return expectedType.cast(bestActions.get(0));
    }

    /* ========================== Minimax recursive ========================== */
    private double minimax(PylosBoard board, PylosGameSimulator sim, int depth, PylosPlayerColor currentTurn, double alpha, double beta) {
        long key = makeBoardKey(board, currentTurn);
        Double cached;
        cached = cache.get(key);
        if (cached != null) return cached;

        // Terminal node check + Monte Carlo fallback
        if (depth >= Math.ceil((double) MAX_DEPTH / (board.getReservesSize(currentTurn) + board.getReservesSize(currentTurn.other()))) || sim.getState() == PylosGameState.COMPLETED
                || sim.getState() == PylosGameState.ABORTED
                || sim.getState() == PylosGameState.DRAW) {

            double eval;
            if (sim.getState() == PylosGameState.MOVE) {
                eval = runMonteCarlo(board, sim, currentTurn);
            } else {
                eval = evaluateBoardCached(board);
            }

            cache.put(key, eval);
            return eval;
        }


        boolean maximizing = (currentTurn == botColor);
        List<Action> actions = generateActionsForState(board, sim.getState(), currentTurn);

        if (actions.isEmpty()) {
            double eval = evaluateBoardCached(board);
            cache.put(key, eval);
            return eval;
        }

        // Sorteer acties voor betere pruning
        actions.sort((a, b) -> Double.compare(actionHeuristic(b), actionHeuristic(a)));

        double best = maximizing ? Double.NEGATIVE_INFINITY : Double.POSITIVE_INFINITY;

        for (Action a : actions) {
            PylosGameState prevState = sim.getState();
            double val;

            if (a instanceof Move m) {
                boolean wasReserve = m.sphere.isReserve();
                PylosLocation oldLoc = wasReserve ? null : m.sphere.getLocation();
                sim.moveSphere(m.sphere, m.location);
                val = minimax(board, sim, depth + 1, nextTurnAfter(prevState, sim.getState(), currentTurn), alpha, beta);
                if (wasReserve) sim.undoAddSphere(m.sphere, prevState, currentTurn);
                else sim.undoMoveSphere(m.sphere, oldLoc, prevState, currentTurn);

            } else if (a instanceof Remove r) {
                if (prevState == PylosGameState.REMOVE_SECOND && r.sphere == null) {
                    sim.pass();
                    val = minimax(board, sim, depth + 1, nextTurnAfter(prevState, sim.getState(), currentTurn), alpha, beta);
                    sim.undoPass(prevState, currentTurn);
                } else {
                    PylosLocation oldLoc = r.sphere.getLocation();
                    sim.removeSphere(r.sphere);
                    val = minimax(board, sim, depth + 1, nextTurnAfter(prevState, sim.getState(), currentTurn), alpha, beta);

                    if (prevState == PylosGameState.REMOVE_FIRST) sim.undoRemoveFirstSphere(r.sphere, oldLoc, prevState, currentTurn);
                    else if (prevState == PylosGameState.REMOVE_SECOND) sim.undoRemoveSecondSphere(r.sphere, oldLoc, prevState, currentTurn);
                }
            } else continue;

            if (maximizing) {
                best = Math.max(best, val);
                alpha = Math.max(alpha, best);
            } else {
                best = Math.min(best, val);
                beta = Math.min(beta, best);
            }

            if (beta <= alpha) break;
        }

        cache.put(key, best);
        return best;
    }

    /* ========================== Actiegeneratie ========================== */

    private List<Action> generateActionsForState(PylosBoard board, PylosGameState state, PylosPlayerColor turn) {
        List<Action> actions = new ArrayList<>();
        switch (state) {
            case MOVE -> {
                // Gebruik alle mogelijke locaties en spheres
                for (PylosLocation location : board.getLocations()) {
                    if (location.isUsable()) {
                        // Reserve spheres
                        if (board.getReservesSize(turn) > 0) {
                            actions.add(new Move(board.getReserve(turn), location));
                        }

                        // Bestaande spheres die kunnen bewegen
                        for (PylosSphere sphere : board.getSpheres(turn)) {
                            if (!sphere.isReserve() && sphere.canMoveTo(location)) {
                                actions.add(new Move(sphere, location));
                            }
                        }

                    }
                }
            }
            case REMOVE_FIRST -> {
                for (PylosSphere s : board.getSpheres(turn)) {
                    if (s.canRemove()) {
                        actions.add(new Remove(s));
                    }
                }
            }
            case REMOVE_SECOND -> {
                for (PylosSphere s : board.getSpheres(turn)) {
                    if (s.canRemove()) {
                        actions.add(new Remove(s));
                    }
                }
                actions.add(new Remove(null)); // pass
            }
            default -> {}
        }
        return actions;
    }

    private PylosPlayerColor nextTurnAfter(PylosGameState prev, PylosGameState now, PylosPlayerColor current) {
        if (prev == PylosGameState.MOVE && now == PylosGameState.REMOVE_FIRST) return current;
        if (prev == PylosGameState.MOVE && now == PylosGameState.MOVE) return current.other();
        if (prev == PylosGameState.REMOVE_FIRST && now == PylosGameState.REMOVE_SECOND) return current;
        if (prev == PylosGameState.REMOVE_SECOND && now == PylosGameState.MOVE) return current.other();
        return current;
    }

    /* ========================== Hulpmethoden ========================== */

    private PylosLocation firstUsable(PylosBoard board) {
        for (PylosLocation loc : board.getLocations()) {
            if (loc.isUsable()) return loc;
        }
        return board.getLocations()[0];
    }

    private double actionHeuristic(Action a) {
        if (a instanceof Move m) {
            // Hogere locaties zijn beter, centrum is beter
            int zScore = m.location.Z * 15;
            int centerDist = Math.abs(m.location.X - 1) + Math.abs(m.location.Y - 1);
            int centerScore = (3 - centerDist) * 5;
            return zScore + centerScore;
        } else if (a instanceof Remove r && r.sphere != null) {
            // Verwijder hogere spheres eerst
            return r.sphere.getLocation().Z * 10;
        }
        return 0;
    }

    private long makeBoardKey(PylosBoard board, PylosPlayerColor turn) {
        long key = 0L;
        int[] layerOffsets = {0,16,25,29};

        for (PylosSphere s : board.getSpheres(PylosPlayerColor.LIGHT)) {
            if (!s.isReserve()) {
                PylosLocation loc = s.getLocation();
                int idx = layerOffsets[loc.Z] + loc.Y * (4 - loc.Z) + loc.X;
                key ^= ZOBRIST[idx][0];
            }
        }
        for (PylosSphere s : board.getSpheres(PylosPlayerColor.DARK)) {
            if (!s.isReserve()) {
                PylosLocation loc = s.getLocation();
                int idx = layerOffsets[loc.Z] + loc.Y * (4 - loc.Z) + loc.X;
                key ^= ZOBRIST[idx][1];
            }
        }

        int lightRes = board.getReservesSize(PylosPlayerColor.LIGHT);
        int darkRes  = board.getReservesSize(PylosPlayerColor.DARK);
        key ^= ZOBRIST_RESERVE_COUNT[0][lightRes];
        key ^= ZOBRIST_RESERVE_COUNT[1][darkRes];
        key ^= ZOBRIST_TURN[turn == PylosPlayerColor.LIGHT ? 0 : 1];
        return key;
    }


    /* ========================== Gecachte Evaluatie ========================== */

    private int evaluateBoardCached(PylosBoard board) {
        long boardKey = makeBoardKey(board, botColor);

        // Check cache eerst
        Integer cached = evalCache.get(boardKey);
        if (cached != null) {
            return cached;
        }

        PylosPlayerColor opponent = botColor.other();
        int myScore = 0;
        int oppScore = 0;

        // Basis scores
        myScore += WEIGHT_RESERVE * board.getReservesSize(botColor);
        oppScore += WEIGHT_RESERVE * board.getReservesSize(opponent);

        // Hoogte, centrum en support
        for (PylosSphere s : board.getSpheres(botColor)) {
            if (!s.isReserve()) {
                PylosLocation loc = s.getLocation();
                myScore += loc.Z * WEIGHT_HEIGHT;
                int distFromCenter = Math.abs(loc.X - 1) + Math.abs(loc.Y - 1);
                myScore += (3 - distFromCenter) * WEIGHT_CENTER;
                if (!loc.getAbove().isEmpty()) myScore += WEIGHT_SUPPORT;
            }
        }

        for (PylosSphere s : board.getSpheres(opponent)) {
            if (!s.isReserve()) {
                PylosLocation loc = s.getLocation();
                oppScore += loc.Z * WEIGHT_HEIGHT;
                int distFromCenter = Math.abs(loc.X - 1) + Math.abs(loc.Y - 1);
                oppScore += (3 - distFromCenter) * WEIGHT_CENTER;
                if (!loc.getAbove().isEmpty()) oppScore += WEIGHT_SUPPORT;
            }
        }

        // Square detection
        PylosSquare[] allSquares = board.getAllSquares();

        for (PylosSquare sq : allSquares) {
            int myInSquare = sq.getInSquare(botColor);
            int oppInSquare = sq.getInSquare(opponent);

            if (myInSquare == 4) {
                myScore += WEIGHT_SQUARE;
            } else if (myInSquare == 3 && oppInSquare == 0) {
                myScore += WEIGHT_ALMOST_SQUARE;
                myScore += WEIGHT_THREAT;
            }

            if (oppInSquare == 4) {
                oppScore += WEIGHT_SQUARE;
            } else if (oppInSquare == 3 && myInSquare == 0) {
                oppScore += WEIGHT_ALMOST_SQUARE;
                myScore -= WEIGHT_THREAT;
            }
        }

        int finalScore = myScore - oppScore;

        // Cache het resultaat
        evalCache.put(boardKey, finalScore);

        return finalScore;
    }

    /* ========================== Monte Carlo rollouts (safe) ========================== */
    private static final class Step {
        final PylosGameState prevState;
        final Action action;
        final boolean wasReserve;
        final PylosLocation oldLoc;
        final PylosPlayerColor colorAtStep;
        Step(PylosGameState prevState, Action action, boolean wasReserve, PylosLocation oldLoc, PylosPlayerColor colorAtStep) {
            this.prevState = prevState;
            this.action = action;
            this.wasReserve = wasReserve;
            this.oldLoc = oldLoc;
            this.colorAtStep = colorAtStep;
        }
    }

    private double runMonteCarlo(PylosBoard board, PylosGameSimulator sim,
                                 PylosPlayerColor startColor) {
        double total = 0.0;

        for (int s = 0; s < SIMULATIONS; s++) {
            rolloutStack.clear();
            PylosPlayerColor turn = startColor;

            int plies = 0;
            while (sim.getState() != PylosGameState.COMPLETED
                    && sim.getState() != PylosGameState.ABORTED
                    && sim.getState() != PylosGameState.DRAW
                    && plies < ROLLOUT_MAX_PLIES) {

                actionBuf.clear();
                fillActionsForState(board, sim.getState(), turn, actionBuf);
                if (actionBuf.isEmpty()) break;

                Action a = pickPolicyAction(actionBuf);
                PylosGameState prev = sim.getState();

                if (a instanceof Move m) {
                    boolean wasReserve = m.sphere.isReserve();
                    PylosLocation oldLoc = wasReserve ? null : m.sphere.getLocation();
                    sim.moveSphere(m.sphere, m.location);
                    rolloutStack.push(new Step(prev, a, wasReserve, oldLoc, turn));

                } else if (a instanceof Remove r) {
                    if (prev == PylosGameState.REMOVE_SECOND && r.sphere == null) {
                        sim.pass();
                        rolloutStack.push(new Step(prev, a, false, null, turn));
                    } else {
                        PylosLocation oldLoc = r.sphere.getLocation();
                        sim.removeSphere(r.sphere);
                        rolloutStack.push(new Step(prev, a, false, oldLoc, turn));
                    }
                }

                PylosGameState now = sim.getState();
                turn = nextTurnAfter(prev, now, turn);
                plies++;
            }

            // Score, then completely unwind to restore original node
            total += evaluateBoardCached(board);

            //hij gaat meerder acties uitvoeren dus we moeten ook het board terugzetten
            while (!rolloutStack.isEmpty()) {
                Step st = rolloutStack.pop();
                if (st.action instanceof Move m) {
                    if (st.wasReserve) sim.undoAddSphere(m.sphere, st.prevState, st.colorAtStep);
                    else sim.undoMoveSphere(m.sphere, st.oldLoc, st.prevState, st.colorAtStep);
                } else if (st.action instanceof Remove r) {
                    if (st.prevState == PylosGameState.REMOVE_SECOND && r.sphere == null) {
                        sim.undoPass(st.prevState, st.colorAtStep);
                    } else if (st.prevState == PylosGameState.REMOVE_FIRST) {
                        sim.undoRemoveFirstSphere(r.sphere, st.oldLoc, st.prevState, st.colorAtStep);
                    } else if (st.prevState == PylosGameState.REMOVE_SECOND) {
                        sim.undoRemoveSecondSphere(r.sphere, st.oldLoc, st.prevState, st.colorAtStep);
                    }
                }
            }
        }
        return total / SIMULATIONS;
    }


    private void fillActionsForState(PylosBoard board, PylosGameState state, PylosPlayerColor turn,
                                     java.util.List<Action> out) {
        switch (state) {
            case MOVE -> {
                for (PylosLocation location : board.getLocations()) {
                    if (location.isUsable()) {
                        if (board.getReservesSize(turn) > 0) {
                            out.add(new Move(board.getReserve(turn), location));
                        }
                        for (PylosSphere sphere : board.getSpheres(turn)) {
                            if (!sphere.isReserve() && sphere.canMoveTo(location)) {
                                out.add(new Move(sphere, location));
                            }
                        }
                    }
                }
            }
            case REMOVE_FIRST -> {
                for (PylosSphere s : board.getSpheres(turn)) {
                    if (s.canRemove()) out.add(new Remove(s));
                }
            }
            case REMOVE_SECOND -> {
                for (PylosSphere s : board.getSpheres(turn)) {
                    if (s.canRemove()) out.add(new Remove(s));
                }
                out.add(new Remove(null)); // pass
            }
            default -> {}
        }
    }
    private Action pickPolicyAction(java.util.List<Action> actions) {
        int n = actions.size();
        if (n == 1) return actions.get(0);

        double rnd = fastRnd.nextDouble();
        // Exploration
        if (rnd < ROLLOUT_EPSILON) {
            int idx = fastRnd.nextInt(n);
            return actions.get(idx);
        }

        // Exploitation: vind top-K door een single pass (geen sortering)
        int k = Math.min(ROLLOUT_TOP_K, n);
        double[] topScore = new double[k];
        int[] topIdx = new int[k];
        for (int i = 0; i < k; i++) { topScore[i] = Double.NEGATIVE_INFINITY; topIdx[i] = -1; }

        for (int i = 0; i < n; i++) {
            double s = actionHeuristic(actions.get(i));
            // vind slechtste in top-K
            int worst = 0;
            for (int t = 1; t < k; t++) if (topScore[t] < topScore[worst]) worst = t;
            if (s > topScore[worst]) { topScore[worst] = s; topIdx[worst] = i; }
        }

        // kies uniform over de collected top-K indices
        int filled = 0;
        for (int t = 0; t < k; t++) if (topIdx[t] >= 0) filled++;
        int pick = fastRnd.nextInt(Math.max(1, filled));
        for (int t = 0; t < k; t++) {
            if (topIdx[t] >= 0) {
                if (pick-- == 0) return actions.get(topIdx[t]);
            }
        }
        // Fallback
        return actions.get(fastRnd.nextInt(n));
    }
}
