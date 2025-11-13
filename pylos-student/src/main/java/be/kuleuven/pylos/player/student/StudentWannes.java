package be.kuleuven.pylos.player.student;

import be.kuleuven.pylos.game.*;
import be.kuleuven.pylos.player.PylosPlayer;

import java.lang.reflect.Field;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.List;

import static java.util.Collections.shuffle;
import java.util.HashMap;
import java.util.Random;

/**
 * Created by Jan on 20/02/2015.
 */
public class StudentWannes extends PylosPlayer {
    private PylosGameSimulator simulator;
    private PylosBoard board;
    private int bestScore;
    private int maxDepth = (PLAYER_COLOR == PylosPlayerColor.LIGHT) ? 6 : 5;
    private int moves = 0;
    PylosSphere bestSphere = null;
    PylosLocation bestLocation = null;

    private static final int MONTE_CARLO_SIMULATIONS = 60;
    private static final int MONTE_CARLO_MAX_PLAYOUT_DEPTH = 35; // Dont go to deep (game usually ends before that)
    private static final int MONTE_CARLO_CANDIDATE_LIMIT = 8;
    private static final double MONTE_CARLO_EPSILON = 1e-6;
    private double DRAW_SCORE = 0.5;
    private static final int LATE_GAME_PLACED_THRESHOLD = 22;
    private static final int LATE_GAME_RESERVE_THRESHOLD = 4;

    private final Random random = new Random();

    // Zobrist hashing components
    private static final long[][][][] zobristTable = new long[4][4][4][2]; // [z][x][y][color]
    private static final long zobristTurnLight;
    private static final long zobristTurnDark;
    private static final Field SIMULATOR_WINNER_FIELD;
    private final HashMap<Long, TranspositionEntry> transpositionTable = new HashMap<>();

    // Transposition table entry
    private static class TranspositionEntry {
        int depth;
        int score;
        int flag; // 0 = exact, 1 = lowerbound (alpha), 2 = upperbound (beta)

        TranspositionEntry(int depth, int score, int flag) {
            this.depth = depth;
            this.score = score;
            this.flag = flag;
        }
    }

    // Initialize Zobrist hash table with random values
    static {
        try {
            SIMULATOR_WINNER_FIELD = PylosGameSimulator.class.getDeclaredField("winner");
            SIMULATOR_WINNER_FIELD.setAccessible(true);
        } catch (NoSuchFieldException e) {
            throw new RuntimeException("Unable to access PylosGameSimulator winner field", e);
        }

        Random rand = new Random(42); // Fixed seed for reproducibility
        for (int z = 0; z < 4; z++) {
            for (int x = 0; x < 4 - z; x++) {
                for (int y = 0; y < 4 - z; y++) {
                    zobristTable[z][x][y][0] = rand.nextLong();
                    zobristTable[z][x][y][1] = rand.nextLong();
                }
            }
        }
        zobristTurnLight = rand.nextLong();
        zobristTurnDark = rand.nextLong();
    }

    // Compute Zobrist hash for the current board state
    private long computeZobristHash() {
        long hash = 0L;
        PylosLocation[] allLocations = board.getLocations();

        for (PylosLocation loc : allLocations) {
            if (loc.isUsed()) {
                PylosSphere sphere = loc.getSphere();
                int colorIndex = sphere.PLAYER_COLOR == PylosPlayerColor.LIGHT ? 0 : 1;
                hash ^= zobristTable[loc.Z][loc.X][loc.Y][colorIndex];
            }
        }

        // XOR with turn indicator
        if (simulator.getColor() == PylosPlayerColor.LIGHT) {
            hash ^= zobristTurnLight;
        } else {
            hash ^= zobristTurnDark;
        }

        return hash;
    }

    private int evaluate() {
        // Use different evaluation for light and dark
        // Needed because the dark player moves second and starts with an advantage
        // White needs to play more aggressively to compensate or try to draw
        if (PLAYER_COLOR == PylosPlayerColor.LIGHT) {
            return evaluate_me_dark();
        } else {
            return evaluate_me_dark();
        }
    }

    private int evaluate_me_light() {
        final PylosPlayerColor my  = PLAYER_COLOR;
        final PylosPlayerColor opp = my.other();

        final int myReserve  = board.getReservesSize(my);
        final int oppReserve = board.getReservesSize(opp);

        final PylosSphere[] mySpheres  = board.getSpheres(my);
        final PylosSphere[] oppSpheres = board.getSpheres(opp);

        /* Phase detection (opening -> endgame) */
        final int placed = board.getNumberOfSpheresOnBoard(); // total on board
        final boolean opening  = placed < 8;
        final boolean midgame  = placed >= 8 && placed <= 20;
        final boolean endgame  = placed > 20;

        /* 1) Mobility (but emphasize *upward* mobility momentum) */
        int myOnBoard = 0, oppOnBoard = 0, myMobile = 0, oppMobile = 0;
        int myUpMoves = 0, oppUpMoves = 0; // how many *legal* lifts to higher Z exist now

        for (PylosSphere s : mySpheres) {
            if (s.isReserve()) continue;
            myOnBoard++;
            if (s.canMove()) myMobile++;
            // count upward destinations for this piece
            for (PylosLocation loc : board.getLocations()) {
                if (s.canMoveTo(loc) && loc.Z > s.getLocation().Z) myUpMoves++;
            }
        }
        for (PylosSphere s : oppSpheres) {
            if (s.isReserve()) continue;
            oppOnBoard++;
            if (s.canMove()) oppMobile++;
            for (PylosLocation loc : board.getLocations()) {
                if (s.canMoveTo(loc) && loc.Z > s.getLocation().Z) oppUpMoves++;
            }
        }

        /* 2) Squares: structure + immediate convertibility */
        int myFull = 0, oppFull = 0;
        int myAlmost = 0, oppAlmost = 0;         // 3 vs 0 in the square
        int myTwoZero = 0, oppTwoZero = 0;       // 2 vs 0 in the square

        int myImmediateCompletions  = 0;         // I can finish a square next move (legal placement exists)
        int oppImmediateCompletions = 0;         // Opp can finish a square next move

        int oppForcingThreats = 0;               // Opp has a 3-open AND at least one legal placement to complete

        for (PylosSquare sq : board.getAllSquares()) {
            int m = sq.getInSquare(my);
            int o = sq.getInSquare(opp);

            if (m == 4) myFull++;
            if (o == 4) oppFull++;

            if (o == 0 && m == 3) myAlmost++;
            if (m == 0 && o == 3) oppAlmost++;

            if (o == 0 && m == 2) myTwoZero++;
            if (m == 0 && o == 2) oppTwoZero++;

            // Detect immediate conversions (any empty location in sq playable by color next move)
            if (m == 3 && o == 0) {
                if (squareHasPlayableHoleFor(sq, my)) myImmediateCompletions++;
            }
            if (o == 3 && m == 0) {
                boolean oppCan = squareHasPlayableHoleFor(sq, opp);
                if (oppCan) {
                    oppImmediateCompletions++;
                    oppForcingThreats++; // this is a must-answer if we can't complete something stronger
                }
            }
        }

        /* 3) Height & top race — keep modest to avoid noisy early overbuilds */
        int[] myLayers = new int[4], oppLayers = new int[4];
        for (PylosSphere s : mySpheres) if (!s.isReserve()) myLayers[s.getLocation().Z]++;
        for (PylosSphere s : oppSpheres) if (!s.isReserve()) oppLayers[s.getLocation().Z]++;

        int myHeightScore  = myLayers[0] + myLayers[1]*2 + myLayers[2]*4 + myLayers[3]*8;
        int oppHeightScore = oppLayers[0] + oppLayers[1]*2 + oppLayers[2]*4 + oppLayers[3]*8;

        int myTopControl = 0, oppTopControl = 0;
        if (board.getAllSquares().length > 0) {
            PylosSquare topSquare = board.getAllSquares()[board.getAllSquares().length - 1];
            myTopControl  = topSquare.getInSquare(my);
            oppTopControl = topSquare.getInSquare(opp);
        }

        /* 4) Efficiency (reusability) */
        double myEff  = myOnBoard  > 0 ? (double) myMobile  / myOnBoard  : 0.0;
        double oppEff = oppOnBoard > 0 ? (double) oppMobile / oppOnBoard : 0.0;
        int efficiencyScore = (int) ((myEff - oppEff) * 100);

        /* =======================  SCORING  ======================= */

        int score = 0;


        /* Reserves: DO NOT lower vs dark; Light collapsing on reserves is a common loss mode. */
        score += (myReserve - oppReserve) * (opening ? 30 : midgame ? 200 : 300);

        /* Mobility baseline */
        score += (myMobile - oppMobile) * (opening ? 10 : midgame ? 50 : 20);

        /* Upward momentum (helps White without reckless early pushes) */
        score += (myUpMoves - oppUpMoves) * (opening ? 2 : midgame ? 8 : 10);

        /* Structure that converts (prioritize almost + immediate completions) */
        score += (myAlmost - oppAlmost) * 180;
        if (simulator.getState() == PylosGameState.MOVE) {
            score += (myImmediateCompletions - oppImmediateCompletions) * 120;  // “can finish next move” is huge
        }
        /* 2-in-0 as controlled potential (slightly higher than dark, but modest) */
        score += (myTwoZero - oppTwoZero) * (opening ? 50 : midgame ? 55 : 45);

        /* Completed squares: historical value; keep small */
        score += (myFull - oppFull) * 20;

        /* Height & top: modest, phase-aware */
        score += (myHeightScore - oppHeightScore) * (opening ? 2 : midgame ? 4 : 7);
        score += (myTopControl - oppTopControl) * (opening ? 10 : midgame ? 20 : 25);

        /* Efficiency: favor reusability (promote-over-reserve bias emerges naturally) */
        score += efficiencyScore;

        /* Forcing threats from opponent: big red flag */
        score -= oppForcingThreats * (opening ? 140 : 160);

        /* Small tempo bonus that decays with development (stabilizes first moves) */
        score += Math.max(0, 70 - placed * 3);

        /* Endgame: tighten on resources & height; mobility matters slightly more */
        if (endgame) {
            score += (myReserve - oppReserve) * 100;
            //score += (myHeightScore - oppHeightScore) * 6;
            score += (myMobile - oppMobile) * 40;
        }

        PylosPlayerColor next = simulator.getColor();
        switch (simulator.getState()) {
            case DRAW -> {
                return 0;
            }
            case MOVE -> {
                if (next == PLAYER_COLOR) {
                    if (oppImmediateCompletions == 1){
                        score += 1000;
                    }
                } else {
                    if (myImmediateCompletions == 1){
                        score -= 1000;
                    }
                }
            }
            case REMOVE_FIRST -> {
                if (next == PLAYER_COLOR) {
                    score += 1000; // Big reward for getting to remove
                } else {
                    score -= 500; // Big penalty if opponent gets to remove
                }
            }
            case REMOVE_SECOND -> {

            }
        }



        return score;
    }

    /** Can 'color' legally play to this location next move (reserve OR a movable sphere)? */
    private boolean canPlayTo(PylosPlayerColor color, PylosLocation loc) {
        if (!loc.isUsable()) return false;
        if (board.getReservesSize(color) > 0) return true; // can drop reserve
        for (PylosSphere s : board.getSpheres(color)) {
            if (!s.isReserve() && s.canMoveTo(loc)) return true;
        }
        return false;
    }

    /** Does the square contain at least one empty hole that 'color' can legally fill next move? */
    private boolean squareHasPlayableHoleFor(PylosSquare sq, PylosPlayerColor color) {
        for (PylosLocation loc : sq.getLocations()) {
            if (!loc.isUsed() && canPlayTo(color, loc)) return true;
        }
        return false;
    }

    private boolean shouldUseMonteCarlo() {
        int placed = board.getNumberOfSpheresOnBoard();
        int myReserve = board.getReservesSize(PLAYER_COLOR);
        int oppReserve = board.getReservesSize(PLAYER_COLOR.other());
        return placed >= LATE_GAME_PLACED_THRESHOLD
                || myReserve <= LATE_GAME_RESERVE_THRESHOLD
                || oppReserve <= LATE_GAME_RESERVE_THRESHOLD;
    }

    private AbstractMap.SimpleEntry<PylosSphere, PylosLocation> selectMonteCarloMove() {
        ArrayList<ScoredMove> moves = sortDoMove();
        if (moves.isEmpty()) {
            return null;
        }

        int limit = Math.min(moves.size(), MONTE_CARLO_CANDIDATE_LIMIT);
        double bestValue = Double.NEGATIVE_INFINITY;
        ArrayList<AbstractMap.SimpleEntry<PylosSphere, PylosLocation>> bestMoves = new ArrayList<>();

        for (int i = 0; i < limit; i++) {
            ScoredMove move = moves.get(i);
            PylosSphere sphere = move.sphere;
            PylosLocation target = move.location;
            PylosLocation prevLoc = move.originalLocation;

            PylosGameState prevState = simulator.getState();
            PylosPlayerColor prevColor = simulator.getColor();
            PylosPlayerColor prevWinner = simulator.getWinner();

            simulator.moveSphere(sphere, target);
            double value = runMonteCarloRollouts(MONTE_CARLO_SIMULATIONS);

            if (prevLoc == null) {
                simulator.undoAddSphere(sphere, prevState, prevColor);
            } else {
                simulator.undoMoveSphere(sphere, prevLoc, prevState, prevColor);
            }
            setSimulatorWinner(prevWinner);

            if (value > bestValue + MONTE_CARLO_EPSILON) {
                bestValue = value;
                bestMoves.clear();
                bestMoves.add(new AbstractMap.SimpleEntry<>(sphere, target));
            } else if (Math.abs(value - bestValue) <= MONTE_CARLO_EPSILON) {
                bestMoves.add(new AbstractMap.SimpleEntry<>(sphere, target));
            }
        }

        if (bestMoves.isEmpty()) {
            return null;
        }
        return bestMoves.get(random.nextInt(bestMoves.size()));
    }

    private PylosSphere selectMonteCarloRemove(boolean firstRemoval) {
        ArrayList<ScoredRemove> removes = sortDoRemove();
        if (removes.isEmpty()) {
            return null;
        }

        int limit = Math.min(removes.size(), MONTE_CARLO_CANDIDATE_LIMIT);
        double bestValue = Double.NEGATIVE_INFINITY;
        ArrayList<PylosSphere> bestChoices = new ArrayList<>();

        for (int i = 0; i < limit; i++) {
            ScoredRemove remove = removes.get(i);
            PylosSphere sphere = remove.sphere;
            PylosLocation prevLoc = sphere.getLocation();

            PylosGameState prevState = simulator.getState();
            PylosPlayerColor prevColor = simulator.getColor();
            PylosPlayerColor prevWinner = simulator.getWinner();

            simulator.removeSphere(sphere);
            double value = runMonteCarloRollouts(MONTE_CARLO_SIMULATIONS);

            if (firstRemoval) {
                simulator.undoRemoveFirstSphere(sphere, prevLoc, prevState, prevColor);
            } else {
                simulator.undoRemoveSecondSphere(sphere, prevLoc, prevState, prevColor);
            }
            setSimulatorWinner(prevWinner);

            if (value > bestValue + MONTE_CARLO_EPSILON) {
                bestValue = value;
                bestChoices.clear();
                bestChoices.add(sphere);
            } else if (Math.abs(value - bestValue) <= MONTE_CARLO_EPSILON) {
                bestChoices.add(sphere);
            }
        }

        if (bestChoices.isEmpty()) {
            return null;
        }
        return bestChoices.get(random.nextInt(bestChoices.size()));
    }

    private PylosSphere selectMonteCarloRemoveOrPass() {
        ArrayList<ScoredRemove> removes = sortDoRemoveOrPass();
        int limit = Math.min(removes.size(), MONTE_CARLO_CANDIDATE_LIMIT);

        double bestValue = Double.NEGATIVE_INFINITY;
        ArrayList<PylosSphere> bestChoices = new ArrayList<>();
        boolean passAdded = false;

        for (int i = 0; i < limit; i++) {
            ScoredRemove remove = removes.get(i);
            PylosSphere sphere = remove.sphere;
            PylosLocation prevLoc = sphere.getLocation();

            PylosGameState prevState = simulator.getState();
            PylosPlayerColor prevColor = simulator.getColor();
            PylosPlayerColor prevWinner = simulator.getWinner();

            simulator.removeSphere(sphere);
            double value = runMonteCarloRollouts(MONTE_CARLO_SIMULATIONS);
            simulator.undoRemoveSecondSphere(sphere, prevLoc, prevState, prevColor);
            setSimulatorWinner(prevWinner);

            if (value > bestValue + MONTE_CARLO_EPSILON) {
                bestValue = value;
                bestChoices.clear();
                bestChoices.add(sphere);
                passAdded = false;
            } else if (Math.abs(value - bestValue) <= MONTE_CARLO_EPSILON) {
                bestChoices.add(sphere);
            }
        }

        PylosGameState prevState = simulator.getState();
        PylosPlayerColor prevColor = simulator.getColor();
        PylosPlayerColor prevWinner = simulator.getWinner();
        simulator.pass();
        double passValue = runMonteCarloRollouts(MONTE_CARLO_SIMULATIONS);
        simulator.undoPass(prevState, prevColor);
        setSimulatorWinner(prevWinner);

        if (passValue > bestValue + MONTE_CARLO_EPSILON) {
            bestValue = passValue;
            bestChoices.clear();
            bestChoices.add(null);
            passAdded = true;
        } else if (Math.abs(passValue - bestValue) <= MONTE_CARLO_EPSILON) {
            if (!passAdded) {
                bestChoices.add(null);
                passAdded = true;
            }
        }

        if (bestChoices.isEmpty()) {
            return null;
        }
        return bestChoices.get(random.nextInt(bestChoices.size()));
    }

    private double runMonteCarloRollouts(int simulations) {
        if (simulations <= 0) {
            return 0.0;
        }
        double total = 0.0;
        PylosPlayerColor baseWinner = simulator.getWinner();
        for (int i = 0; i < simulations; i++) {
            total += simulateRandomPlayout();
            setSimulatorWinner(baseWinner);
        }
        return total / simulations;
    }

    private double simulateRandomPlayout() {
        ArrayList<SimulationStep> history = new ArrayList<>();
        double result = DRAW_SCORE;
        boolean finished = false;
        int steps = 0;

        while (!finished && steps < MONTE_CARLO_MAX_PLAYOUT_DEPTH) {
            PylosPlayerColor winner = simulator.getWinner();
            if (winner != null) {
                result = winner == PLAYER_COLOR ? 1.0 : 0.0;
                finished = true;
                break;
            }
            if (simulator.getState() == PylosGameState.DRAW) {
                result = DRAW_SCORE;
                finished = true;
                break;
            }

            PylosGameState state = simulator.getState();
            switch (state) {
                case MOVE -> {
                    ArrayList<ScoredMove> moves = sortDoMove();
                    if (moves.isEmpty()) {
                        result = evaluationToProbability(evaluate());
                        finished = true;
                    } else {
                        ScoredMove choice = moves.get(random.nextInt(moves.size()));
                        PylosSphere sphere = choice.sphere;
                        PylosLocation prevLoc = choice.originalLocation;
                        PylosGameState prevState = simulator.getState();
                        PylosPlayerColor prevColor = simulator.getColor();
                        PylosPlayerColor prevWinner = simulator.getWinner();
                        simulator.moveSphere(sphere, choice.location);
                        history.add(new SimulationStep(SimulationActionType.MOVE, sphere, prevLoc, prevState, prevColor, prevWinner));
                    }
                }
                case REMOVE_FIRST -> {
                    ArrayList<ScoredRemove> removes = sortDoRemove();
                    if (removes.isEmpty()) {
                        result = evaluationToProbability(evaluate());
                        finished = true;
                    } else {
                        ScoredRemove choice = removes.get(random.nextInt(removes.size()));
                        PylosSphere sphere = choice.sphere;
                        PylosLocation prevLoc = sphere.getLocation();
                        PylosGameState prevState = simulator.getState();
                        PylosPlayerColor prevColor = simulator.getColor();
                        PylosPlayerColor prevWinner = simulator.getWinner();
                        simulator.removeSphere(sphere);
                        history.add(new SimulationStep(SimulationActionType.REMOVE_FIRST, sphere, prevLoc, prevState, prevColor, prevWinner));
                    }
                }
                case REMOVE_SECOND -> {
                    ArrayList<ScoredRemove> removes = sortDoRemoveOrPass();
                    int options = removes.size() + 1; // +1 for pass
                    int pick = random.nextInt(Math.max(1, options));
                    if (pick < removes.size()) {
                        ScoredRemove choice = removes.get(pick);
                        PylosSphere sphere = choice.sphere;
                        PylosLocation prevLoc = sphere.getLocation();
                        PylosGameState prevState = simulator.getState();
                        PylosPlayerColor prevColor = simulator.getColor();
                        PylosPlayerColor prevWinner = simulator.getWinner();
                        simulator.removeSphere(sphere);
                        history.add(new SimulationStep(SimulationActionType.REMOVE_SECOND, sphere, prevLoc, prevState, prevColor, prevWinner));
                    } else {
                        PylosGameState prevState = simulator.getState();
                        PylosPlayerColor prevColor = simulator.getColor();
                        PylosPlayerColor prevWinner = simulator.getWinner();
                        simulator.pass();
                        history.add(new SimulationStep(SimulationActionType.PASS, null, null, prevState, prevColor, prevWinner));
                    }
                }
                default -> {
                    result = evaluationToProbability(evaluate());
                    finished = true;
                }
            }

            if (!finished) {
                steps++;
            }
        }

        if (!finished) {
            result = evaluationToProbability(evaluate());
        }

        for (int i = history.size() - 1; i >= 0; i--) {
            undoSimulationStep(history.get(i));
        }

        return result;
    }

    private void undoSimulationStep(SimulationStep step) {
        switch (step.type) {
            case MOVE -> {
                if (step.previousLocation == null) {
                    simulator.undoAddSphere(step.sphere, step.prevState, step.prevColor);
                } else {
                    simulator.undoMoveSphere(step.sphere, step.previousLocation, step.prevState, step.prevColor);
                }
            }
            case REMOVE_FIRST -> simulator.undoRemoveFirstSphere(step.sphere, step.previousLocation, step.prevState, step.prevColor);
            case REMOVE_SECOND -> simulator.undoRemoveSecondSphere(step.sphere, step.previousLocation, step.prevState, step.prevColor);
            case PASS -> simulator.undoPass(step.prevState, step.prevColor);
        }
        setSimulatorWinner(step.prevWinner);
    }

    private double evaluationToProbability(int eval) {
        double scaled = Math.tanh(eval / 600.0);
        double value = 0.5 + 0.5 * scaled;
        if (value < 0.0) {
            return 0.0;
        }
        if (value > 1.0) {
            return 1.0;
        }
        return value;
    }

    private void setSimulatorWinner(PylosPlayerColor winner) {
        try {
            SIMULATOR_WINNER_FIELD.set(simulator, winner);
        } catch (IllegalAccessException e) {
            throw new IllegalStateException("Unable to reset simulator winner state", e);
        }
    }

    private enum SimulationActionType {
        MOVE,
        REMOVE_FIRST,
        REMOVE_SECOND,
        PASS
    }

    private static class SimulationStep {
        final SimulationActionType type;
        final PylosSphere sphere;
        final PylosLocation previousLocation;
        final PylosGameState prevState;
        final PylosPlayerColor prevColor;
        final PylosPlayerColor prevWinner;

        SimulationStep(SimulationActionType type, PylosSphere sphere, PylosLocation previousLocation,
                       PylosGameState prevState, PylosPlayerColor prevColor, PylosPlayerColor prevWinner) {
            this.type = type;
            this.sphere = sphere;
            this.previousLocation = previousLocation;
            this.prevState = prevState;
            this.prevColor = prevColor;
            this.prevWinner = prevWinner;
        }
    }




    private int evaluate_me_dark() {
        PylosPlayerColor my = PLAYER_COLOR;
        PylosPlayerColor opp = my.other();

        int myReserve = board.getReservesSize(PLAYER_COLOR);
        int oppReserve = board.getReservesSize(PLAYER_COLOR.other());

        PylosSphere[] mySpheres = board.getSpheres(my);
        PylosSphere[] oppSpheres = board.getSpheres(opp);

        // 1. Mobility: pieces that can be promoted
        int myMobile = 0;
        int oppMobile = 0;
        int myOnBoard = 0;
        int oppOnBoard = 0;

        for (PylosSphere s : mySpheres) {
            if (s.isReserve()) continue;
            myOnBoard++;
            if (s.canMove()) myMobile++;
        }
        for (PylosSphere s : oppSpheres) {
            if (s.isReserve()) continue;
            oppOnBoard++;
            if (s.canMove()) oppMobile++;
        }

        // 2. Square control with more granular scoring
        int myFull = 0, oppFull = 0;
        int myAlmost = 0, oppAlmost = 0;  // 3 of mine, 0 opponent
        int myTwoZero = 0, oppTwoZero = 0; // 2 of mine, 0 opponent
        int myThreats = 0, oppThreats = 0; // 3 of opponent, can be blocked

        for (PylosSquare sq : board.getAllSquares()) {
            int m = sq.getInSquare(my);
            int o = sq.getInSquare(opp);

            if (m == 4) myFull++;
            if (o == 4) oppFull++;

            if (o == 0 && m == 3) myAlmost++;
            if (m == 0 && o == 3) oppAlmost++;

            if (o == 0 && m == 2) myTwoZero++;
            if (m == 0 && o == 2) oppTwoZero++;

            // Defensive: count opponent threats we need to block
            if (o == 3 && m > 0) myThreats++; // We can block
            if (m == 3 && o > 0) oppThreats++; // Opponent can block
        }

        // 3. Height advantage: count spheres on each level
        int[] myLayers = new int[4];
        int[] oppLayers = new int[4];

        for (PylosSphere s : mySpheres) {
            if (!s.isReserve()) {
                myLayers[s.getLocation().Z]++;
            }
        }
        for (PylosSphere s : oppSpheres) {
            if (!s.isReserve()) {
                oppLayers[s.getLocation().Z]++;
            }
        }

        // Weight higher layers more (getting closer to winning)
        int myHeightScore = myLayers[0] + myLayers[1] * 2 + myLayers[2] * 4 + myLayers[3] * 8;
        int oppHeightScore = oppLayers[0] + oppLayers[1] * 2 + oppLayers[2] * 4 + oppLayers[3] * 8;

        // 4. Control of top square (winning condition area)
        int myTopControl = 0, oppTopControl = 0;
        if (board.getAllSquares().length > 0) {
            PylosSquare topSquare = board.getAllSquares()[board.getAllSquares().length - 1];
            myTopControl = topSquare.getInSquare(my);
            oppTopControl = topSquare.getInSquare(opp);
        }

        // 5. Efficiency: ratio of mobile to on-board pieces
        double myEfficiency = myOnBoard > 0 ? (double) myMobile / myOnBoard : 0;
        double oppEfficiency = oppOnBoard > 0 ? (double) oppMobile / oppOnBoard : 0;
        int efficiencyScore = (int)((myEfficiency - oppEfficiency) * 100);

        // Composite score with carefully tuned weights
        int score = 0;

        // Reserve difference is CRITICAL - running out loses the game
        score += (myReserve - oppReserve) * 150;

        // Mobility is valuable for future flexibility
        score += (myMobile - oppMobile) * 45;

        // Almost-complete squares are very valuable (opportunity to remove)
        score += (myAlmost - oppAlmost) * 180;

        // Two-in-a-row on empty squares (potential)
        score += (myTwoZero - oppTwoZero) * 40;

        // Completed squares (historical, less valuable than future opportunities)
        score += (myFull - oppFull) * 25;

        // Height advantage (progress toward winning)
//        score += (myHeightScore - oppHeightScore) * 8;

        // Top square control (critical endgame)
        score += (myTopControl - oppTopControl) * 30;

        // Efficiency (ability to reuse pieces)
        score += efficiencyScore;

        // Defensive: penalize opponent threats
        score += (myThreats - oppThreats) * 60;

        // Endgame heuristic: if reserves are low, prioritize differently
        int totalReserves = myReserve + oppReserve;
        if (totalReserves < 8) {
            // In endgame, mobility and height become more important
            score += (myMobile - oppMobile) * 30; // Bonus
            score += (myHeightScore - oppHeightScore) * 5; // Bonus

            // Reserve difference becomes even more critical
            score += (myReserve - oppReserve) * 50; // Additional bonus
        }

        return score;
    }


    private int minimax(int depth, int alpha, int beta) {
        PylosPlayerColor winner = this.simulator.getWinner();
        PylosGameState state = simulator.getState();

        if (winner != null) {
            if (winner == PLAYER_COLOR) return 50000 - depth;
            return -50000 + depth;
        }

        if (state == PylosGameState.DRAW) {
            return 0;
        }

        if (depth == 0) return evaluate();

        // Compute Zobrist hash for this position
        long zobristHash = computeZobristHash();

        // Check transposition table
        TranspositionEntry ttEntry = transpositionTable.get(zobristHash);
        if (ttEntry != null && ttEntry.depth >= depth) {
            if (ttEntry.flag == 0) { // Exact score
                return ttEntry.score;
            } else if (ttEntry.flag == 1) { // Lower bound
                alpha = Math.max(alpha, ttEntry.score);
            } else if (ttEntry.flag == 2) { // Upper bound
                beta = Math.min(beta, ttEntry.score);
            }
            if (alpha >= beta) {
                return ttEntry.score;
            }
        }

        boolean isMaxNode = (this.simulator.getColor() == PLAYER_COLOR);
        int bestScore = isMaxNode ? Integer.MIN_VALUE : Integer.MAX_VALUE;
        int originalAlpha = alpha;

        PylosGameState prevState = this.simulator.getState();
        PylosPlayerColor prevColor = this.simulator.getColor();

        switch (state) {
            case MOVE -> {
                // Use move sorting for better alpha-beta pruning
                ArrayList<ScoredMove> sortedMoves = sortDoMove();

                for (ScoredMove move : sortedMoves) {
                    PylosSphere sphere = move.sphere;
                    PylosLocation loc = move.location;
                    PylosLocation prevLoc = move.originalLocation;

                    this.simulator.moveSphere(sphere, loc);
                    int score = minimax(depth - 1, alpha, beta);

                    if (prevLoc == null) this.simulator.undoAddSphere(sphere, prevState, prevColor);
                    else this.simulator.undoMoveSphere(sphere, prevLoc, prevState, prevColor);

                    if (isMaxNode) {
                        bestScore = Math.max(bestScore, score);
                        alpha = Math.max(alpha, bestScore);
                    } else {
                        bestScore = Math.min(bestScore, score);
                        beta = Math.min(beta, bestScore);
                    }

                    if (beta <= alpha) return bestScore; // Prune
                }
            }

            case REMOVE_FIRST -> {
                // Use move sorting for better alpha-beta pruning
                ArrayList<ScoredRemove> sortedRemoves = sortDoRemove();

                for (ScoredRemove remove : sortedRemoves) {
                    PylosSphere sphere = remove.sphere;
                    PylosLocation prevLoc = sphere.getLocation();
                    simulator.removeSphere(sphere);
                    int score = minimax(depth - 1, alpha, beta);
                    simulator.undoRemoveFirstSphere(sphere, prevLoc, prevState, prevColor);

                    if (isMaxNode) {
                        bestScore = Math.max(bestScore, score);
                        alpha = Math.max(alpha, bestScore);
                    } else {
                        bestScore = Math.min(bestScore, score);
                        beta = Math.min(beta, bestScore);
                    }

                    if (beta <= alpha) return bestScore; // prune
                }
            }

            case REMOVE_SECOND -> {
                // Use move sorting for better alpha-beta pruning
                ArrayList<ScoredRemove> sortedRemoves = sortDoRemoveOrPass();

                for (ScoredRemove remove : sortedRemoves) {
                    PylosSphere sphere = remove.sphere;
                    PylosLocation prevLoc = sphere.getLocation();
                    simulator.removeSphere(sphere);
                    int score = minimax(depth - 1, alpha, beta);
                    simulator.undoRemoveSecondSphere(sphere, prevLoc, prevState, prevColor);

                    if (isMaxNode) {
                        bestScore = Math.max(bestScore, score);
                        alpha = Math.max(alpha, bestScore);
                    } else {
                        bestScore = Math.min(bestScore, score);
                        beta = Math.min(beta, bestScore);
                    }

                    if (beta <= alpha) return bestScore; // Alpha-beta cutoff
                }

                // Always evaluate pass as the last option
                this.simulator.pass();
                int score = minimax(depth - 1, alpha, beta);
                this.simulator.undoPass(prevState, prevColor);

                if (isMaxNode) {
                    bestScore = Math.max(bestScore, score);
                } else {
                    bestScore = Math.min(bestScore, score);
                }
            }
        }

        // Store in transposition table
        int flag;
        if (bestScore <= originalAlpha) {
            flag = 2; // Upper bound
        } else if (bestScore >= beta) {
            flag = 1; // Lower bound
        } else {
            flag = 0; // Exact
        }
        transpositionTable.put(zobristHash, new TranspositionEntry(depth, bestScore, flag));

        return bestScore;
    }

    public class ScoredMove implements Comparable<ScoredMove> {
        PylosSphere sphere;
        PylosLocation location;
        PylosLocation originalLocation; // null if from reserve
        int heuristicScore;

        ScoredMove(PylosSphere sphere, PylosLocation location, PylosLocation originalLocation) {
            this.sphere = sphere;
            this.location = location;
            this.originalLocation = originalLocation;
            this.heuristicScore = scoreMoveHeuristic(location, originalLocation);
        }


        private int scoreMoveHeuristic(PylosLocation location, PylosLocation originalLocation) {
            int score = 0;
            PylosPlayerColor currentPlayer = simulator.getColor();

            // 1. Prioritize higher layers (getting closer to winning)
            score += location.Z * 200;

            // 2. Check if this move creates or nearly creates a square
            for (PylosSquare sq : location.getSquares()) {
                int myCount = sq.getInSquare(currentPlayer);
                if (myCount == 3) {
                    score += 500; // About to complete a square - very valuable!
                } else if (myCount == 2) {
                    score += 150; // Two in a row - good potential
                } else if (myCount == 1) {
                    score += 30;
                }
            }

            // 3. Prefer promotion moves (reusing pieces) over reserve moves
            if (originalLocation != null) {
                score += 100; // Promotion is generally better than using reserve
            }

            // 4. On bottom layer, prefer more central positions (stronger foundation)
            if (location.Z == 0) {
                int centerDist = Math.abs(location.X - 1) + Math.abs(location.Y - 1);
                score += (4 - centerDist) * 15;
            }

            // 5. Penalize moves that block our own future promotion opportunities
            if (originalLocation != null && originalLocation.Z < location.Z) {
                score += 50; // Reward upward moves
            }

            return score;
        }

        @Override
        public int compareTo(ScoredMove other) {
            return Integer.compare(other.heuristicScore, this.heuristicScore); // Descending order
        }
    }

    public class ScoredRemove implements Comparable<ScoredRemove> {
        PylosSphere sphere;
        int heuristicScore;

        ScoredRemove(PylosSphere sphere) {
            this.sphere = sphere;
            this.heuristicScore = scoreRemoveHeuristic(sphere);
        }

        private int scoreRemoveHeuristic(PylosSphere sphere) {
            int score = 0;
            PylosLocation loc = sphere.getLocation();

            // 1. Prioritize removing from lower layers (more valuable to reclaim)
            score += (3 - loc.Z) * 150;

            // 2. Prefer removing isolated pieces (less strategic impact)
            int involvedInSquares = 0;
            for (PylosSquare sq : loc.getSquares()) {
                int myCount = sq.getInSquare(PLAYER_COLOR);
                if (myCount >= 2) {
                    involvedInSquares++;
                }
            }
            score += (4 - involvedInSquares) * 80; // Prefer pieces not involved in potential squares

            // 3. On bottom layer, prefer edge pieces over center
            if (loc.Z == 0) {
                int centerDist = Math.abs(loc.X - 1) + Math.abs(loc.Y - 1);
                score += centerDist * 20; // Higher score for pieces farther from center
            }

            return score;
        }

        @Override
        public int compareTo(ScoredRemove other) {
            return Integer.compare(other.heuristicScore, this.heuristicScore); // Descending order
        }
    }

    private void init(PylosGameState state,  PylosBoard board) {
        this.simulator = new PylosGameSimulator(state, PLAYER_COLOR, board);
        this.board = board;
        if (board.getNumberOfSpheresOnBoard() < 2) {
            this.moves = 0;
            this.DRAW_SCORE = (PLAYER_COLOR == PylosPlayerColor.LIGHT) ? 0.55 : 0.45;
        }
        this.maxDepth = (PLAYER_COLOR == PylosPlayerColor.LIGHT) ? 6 : 5;
        this.bestSphere = null;
        this.bestLocation = null;
        this.bestScore = Integer.MIN_VALUE;
        if (true){//(simulator.getState() != PylosGameState.REMOVE_SECOND) {
            this.transpositionTable.clear(); // Clear transposition table for new move
        }
        this.moves++;
        if (simulator.getState() == PylosGameState.REMOVE_FIRST) {
            //this.maxDepth += 3;
        }
    }


    public ArrayList<ScoredMove> sortDoMove() {
        ArrayList<ScoredMove> moves = new ArrayList<>();
        PylosLocation[] allLocations = this.board.getLocations();
        PylosSphere[] mySpheres = this.board.getSpheres(this.simulator.getColor());
        PylosSphere myReserveSphere = this.board.getReserve(this.simulator.getColor());

        // Collect moves from reserve
        if (myReserveSphere != null) {
            for (PylosLocation loc : allLocations) {
                if (loc.isUsable()) {
                    moves.add(new ScoredMove(myReserveSphere, loc, null));
                }
            }
        }

        // Collect promotion moves (moving existing pieces up)
        for (PylosSphere sphere : mySpheres) {
            if (!sphere.isReserve() && sphere.canMove()) {
                for (PylosLocation loc : allLocations) {
                    if (sphere.canMoveTo(loc)) {
                        moves.add(new ScoredMove(sphere, loc, sphere.getLocation()));
                    }
                }
            }
        }

        moves.sort(null); // Uses Comparable implementation
        return moves;
    }


    public ArrayList<ScoredRemove> sortDoRemove() {
        ArrayList<ScoredRemove> removes = new ArrayList<>();
        PylosSphere[] mySpheres = this.board.getSpheres(this.simulator.getColor());

        for (PylosSphere sphere : mySpheres) {
            if (!sphere.isReserve() && sphere.canRemove()) {
                removes.add(new ScoredRemove(sphere));
            }
        }

        removes.sort(null); // Uses Comparable implementation
        return removes;
    }

    public ArrayList<ScoredRemove> sortDoRemoveOrPass() {
        return sortDoRemove();
    }

    @Override
    public void doMove(PylosGameIF game, PylosBoard board) {
        init(game.getState(), board);
        PylosGameState prevState = game.getState();
        PylosPlayerColor prevColor = PLAYER_COLOR;

        if (shouldUseMonteCarlo()) {
            AbstractMap.SimpleEntry<PylosSphere, PylosLocation> monteCarloMove = selectMonteCarloMove();
            if (monteCarloMove != null) {
                game.moveSphere(monteCarloMove.getKey(), monteCarloMove.getValue());
                return;
            }
        }

        // USE SORTED MOVES for better alpha-beta pruning at root
        ArrayList<ScoredMove> sortedMoves = sortDoMove();
        ArrayList<AbstractMap.SimpleEntry<PylosSphere, PylosLocation>> bestMoves = new ArrayList<>();

        for (ScoredMove move : sortedMoves) {
            PylosSphere sphere = move.sphere;
            PylosLocation loc = move.location;
            PylosLocation prevLoc = move.originalLocation;

            this.simulator.moveSphere(sphere, loc);
            int score = minimax(this.maxDepth, -50000, 50000);
            if (prevLoc == null) {
                // Move from reserve
                this.simulator.undoAddSphere(sphere, prevState, prevColor);
                score -= 1; // Slight penalty to prefer promotion moves

            } else {
                // Promotion move
                this.simulator.undoMoveSphere(sphere, prevLoc, prevState, prevColor);

            }
            if (score > this.bestScore) {
                this.bestScore = score;
                this.bestLocation = loc;
                this.bestSphere = sphere;
                bestMoves.clear();
                bestMoves.add(new AbstractMap.SimpleEntry<>(sphere, loc));
            } else if (score == this.bestScore) {
                bestMoves.add(new AbstractMap.SimpleEntry<>(sphere, loc));
            }
        }
        shuffle(bestMoves);
        AbstractMap.SimpleEntry<PylosSphere, PylosLocation> toPlay = bestMoves.get(0);
        game.moveSphere(toPlay.getKey(), toPlay.getValue());

    }


    @Override
    public void doRemove(PylosGameIF game, PylosBoard board) {
        init(game.getState(), board);

        PylosGameState prevState = game.getState();
        PylosPlayerColor prevColor = PLAYER_COLOR;

        if (shouldUseMonteCarlo()) {
            PylosSphere monteCarloRemove = selectMonteCarloRemove(true);
            if (monteCarloRemove != null) {
                game.removeSphere(monteCarloRemove);
                return;
            }
        }

        // USE SORTED REMOVES for better alpha-beta pruning at root
        ArrayList<ScoredRemove> sortedRemoves = sortDoRemove();
        ArrayList<PylosSphere> bestRemoves = new ArrayList<>();

        for (ScoredRemove remove : sortedRemoves) {
            PylosSphere sphere = remove.sphere;
            PylosLocation prevLoc = sphere.getLocation();

            simulator.removeSphere(sphere);
            int score = minimax(this.maxDepth, -50000, 50000);
            simulator.undoRemoveFirstSphere(sphere, prevLoc, prevState, prevColor);

            if (score > this.bestScore) {
                this.bestScore = score;
                this.bestLocation = null;
                this.bestSphere = sphere;
                bestRemoves.clear();
                bestRemoves.add(sphere);
            } else if (score == this.bestScore) {
                bestRemoves.add(sphere);
            }
        }
        shuffle(bestRemoves);
        PylosSphere toRemove = bestRemoves.get(0);
        game.removeSphere(toRemove);
    }

    @Override
    public void doRemoveOrPass(PylosGameIF game, PylosBoard board) {
        init(game.getState(), board);

        PylosGameState prevState = game.getState();
        PylosPlayerColor prevColor = PLAYER_COLOR;

        if (shouldUseMonteCarlo()) {
            PylosSphere decision = selectMonteCarloRemoveOrPass();
            if (decision == null) {
                game.pass();
            } else {
                game.removeSphere(decision);
            }
            return;
        }

        // USE SORTED REMOVES for better alpha-beta pruning at root
        ArrayList<ScoredRemove> sortedRemoves = sortDoRemoveOrPass();
        ArrayList<PylosSphere> bestRemoves = new ArrayList<>();


        for (ScoredRemove remove : sortedRemoves) {
            PylosSphere sphere = remove.sphere;
            PylosLocation prevLoc = sphere.getLocation();

            simulator.removeSphere(sphere);
            int score = minimax(this.maxDepth, -50000, 50000);
            simulator.undoRemoveSecondSphere(sphere, prevLoc, prevState, prevColor);

            if (score > this.bestScore) {
                this.bestScore = score;
                this.bestLocation = null;
                this.bestSphere = sphere;
                bestRemoves.clear();
                bestRemoves.add(sphere);
            } else if (score == this.bestScore) {
                bestRemoves.add(sphere);
            }
        }

        // Evaluate pass as the last option
        this.simulator.pass();
        int passScore = minimax(this.maxDepth, -50000, 50000);
        this.simulator.undoPass(prevState, prevColor);

        if (passScore > this.bestScore) {
            game.pass();
        } else {
            if (passScore == this.bestScore) {
                bestRemoves.add(null); // Representing pass
            }
            shuffle(bestRemoves);
            PylosSphere toRemove = bestRemoves.get(0);
            if (toRemove == null) {
                game.pass();
            } else {
                game.removeSphere(toRemove);
            }
        }
    }
}
