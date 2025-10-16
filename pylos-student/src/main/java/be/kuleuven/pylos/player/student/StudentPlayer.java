package be.kuleuven.pylos.player.student;

import be.kuleuven.pylos.game.*;
import be.kuleuven.pylos.player.PylosPlayer;
import java.util.HashMap;
import java.util.Map;

public class StudentPlayer extends PylosPlayer {
    private PylosGameSimulator simulator;
    private PylosBoard board;
    private double bestMinimax;
    private PylosSphere bestSphere;
    private PylosLocation bestLocation;
    private int branchDepth;
    private final Map<Long, Double> evaluationCache = new HashMap<>();

    private static final double WIN_THIS = 2000;
    private static final double WIN_OTHER = -2000;
    private static final double INITIAL_THIS = -9999;
    private static final double INITIAL_OTHER = 9999;
    private static final int MAX_BRANCH_DEPTH = 6;

    @Override
    public void doMove(PylosGameIF game, PylosBoard board) {
        init(game.getState(), board);
        PylosSphere myReserveSphere = board.getReserve(this);
        PylosSphere[] mySpheres = board.getSpheres(this);
        PylosLocation[] locations = board.getLocations();

        if (myReserveSphere != null) {
            for (PylosSphere sphere : mySpheres) {
                if (!sphere.isReserve()) {
                    for (PylosLocation location : locations) {
                        if (sphere.canMoveTo(location)) {
                            PylosLocation prevLocation = sphere.getLocation();
                            simulator.moveSphere(sphere, location);
                            double minimax = branchStep(bestMinimax, INITIAL_OTHER);
                            eval(minimax, sphere, location);
                            simulator.undoMoveSphere(sphere, prevLocation, PylosGameState.MOVE, PLAYER_COLOR);
                        }
                    }
                }
            }
        }

        if (myReserveSphere != null) {
            for (PylosLocation location : locations) {
                if (location.isUsable()) {
                    simulator.moveSphere(myReserveSphere, location);
                    double minimax = branchStep(bestMinimax, INITIAL_OTHER);
                    eval(minimax, myReserveSphere, location);
                    simulator.undoAddSphere(myReserveSphere, PylosGameState.MOVE, PLAYER_COLOR);
                }
            }
        }
        game.moveSphere(bestSphere, bestLocation);
    }

    @Override
    public void doRemove(PylosGameIF game, PylosBoard board) {
        init(game.getState(), board);
        for (PylosSphere sphere : board.getSpheres(PLAYER_COLOR)) {
            if (sphere.canRemove()) {
                tryRemove(sphere, PylosGameState.REMOVE_FIRST);
            }
        }
        game.removeSphere(bestSphere);
    }

    @Override
    public void doRemoveOrPass(PylosGameIF game, PylosBoard board) {
        init(game.getState(), board);
        for (PylosSphere sphere : board.getSpheres(PLAYER_COLOR)) {
            if (sphere.canRemove()) {
                tryRemove(sphere, PylosGameState.REMOVE_SECOND);
            }
        }
        simulator.pass();
        double minimax = branchStep(bestMinimax, INITIAL_OTHER);
        eval(minimax, null, null);
        simulator.undoPass(PylosGameState.REMOVE_SECOND, PLAYER_COLOR);
        if (bestSphere != null) {
            game.removeSphere(bestSphere);
        } else {
            game.pass();
        }
    }

    private void init(PylosGameState state, PylosBoard board) {
        this.simulator = new PylosGameSimulator(state, PLAYER_COLOR, board);
        this.board = board;
        this.bestMinimax = INITIAL_THIS;
        this.bestSphere = null;
        this.bestLocation = null;
        this.branchDepth = 0;
        this.evaluationCache.clear();
    }

    private void eval(double minimax, PylosSphere sphere, PylosLocation location) {
        if (minimax > bestMinimax) {
            bestMinimax = minimax;
            bestSphere = sphere;
            bestLocation = location;
        }
    }

    private void tryRemove(PylosSphere sphere, PylosGameState state) {
        PylosLocation prevLocation = sphere.getLocation();
        simulator.removeSphere(sphere);
        double minimax = branchStep(bestMinimax, INITIAL_OTHER);
        eval(minimax, sphere, null);
        if (state == PylosGameState.REMOVE_FIRST) {
            simulator.undoRemoveFirstSphere(sphere, prevLocation, state, PLAYER_COLOR);
        } else {
            simulator.undoRemoveSecondSphere(sphere, prevLocation, state, PLAYER_COLOR);
        }
    }

    private double branchStep(double alpha, double beta) {
        if (branchDepth == MAX_BRANCH_DEPTH || simulator.getState() == PylosGameState.COMPLETED || simulator.getState() == PylosGameState.DRAW) {
            if (simulator.getState() == PylosGameState.COMPLETED) {
                double result = (simulator.getWinner() == PLAYER_COLOR ? WIN_THIS : WIN_OTHER);
                return result + (simulator.getWinner() == PLAYER_COLOR ? -branchDepth : branchDepth);
            } else if (simulator.getState() == PylosGameState.DRAW) {
                return 0;
            }
            return evaluatePosition();
        }
        PylosGameState state = simulator.getState();
        branchDepth++;
        double result;
        switch (state) {
            case MOVE: result = branchDoMove(alpha, beta); break;
            case REMOVE_FIRST: result = branchDoRemove(alpha, beta); break;
            case REMOVE_SECOND: result = branchDoRemoveOrPass(alpha, beta); break;
            default: throw new IllegalStateException("Unexpected game state: " + state);
        }
        branchDepth--;
        return result;
    }

    private double branchDoMove(double alpha, double beta) {
        PylosPlayerColor color = simulator.getColor();
        boolean isMaximizing = (color == PLAYER_COLOR);
        double minimax = isMaximizing ? INITIAL_THIS : INITIAL_OTHER;
        PylosSphere reserve = board.getReserve(color);
        PylosSphere[] spheres = board.getSpheres(color);
        PylosLocation[] locations = board.getLocations();

        if (reserve != null) {
            for (PylosSphere sphere : spheres) {
                if (!sphere.isReserve()) {
                    for (PylosLocation loc : locations) {
                        if (sphere.canMoveTo(loc)) {
                            PylosLocation prev = sphere.getLocation();
                            simulator.moveSphere(sphere, loc);
                            double result = branchStep(alpha, beta);
                            minimax = updateMinimax(minimax, result, isMaximizing);
                            simulator.undoMoveSphere(sphere, prev, PylosGameState.MOVE, color);
                            if (isMaximizing) alpha = Math.max(alpha, result); else beta = Math.min(beta, result);
                            if (alpha >= beta) return minimax;
                        }
                    }
                }
            }
        }
        if (reserve != null) {
            for (PylosLocation loc : locations) {
                if (loc.isUsable()) {
                    simulator.moveSphere(reserve, loc);
                    double result = branchStep(alpha, beta);
                    minimax = updateMinimax(minimax, result, isMaximizing);
                    simulator.undoAddSphere(reserve, PylosGameState.MOVE, color);
                    if (isMaximizing) alpha = Math.max(alpha, result); else beta = Math.min(beta, result);
                    if (alpha >= beta) return minimax;
                }
            }
        }
        if (minimax == (isMaximizing ? INITIAL_THIS : INITIAL_OTHER)) return evaluatePosition();
        return minimax;
    }

    private double branchDoRemove(double alpha, double beta) {
        PylosPlayerColor color = simulator.getColor();
        boolean isMaximizing = (color == PLAYER_COLOR);
        double minimax = isMaximizing ? INITIAL_THIS : INITIAL_OTHER;
        for (PylosSphere sphere : board.getSpheres(color)) {
            if (sphere.canRemove()) {
                PylosLocation prev = sphere.getLocation();
                simulator.removeSphere(sphere);
                double result = branchStep(alpha, beta);
                minimax = updateMinimax(minimax, result, isMaximizing);
                simulator.undoRemoveFirstSphere(sphere, prev, PylosGameState.REMOVE_FIRST, color);
                if (isMaximizing) alpha = Math.max(alpha, result); else beta = Math.min(beta, result);
                if (alpha >= beta) return minimax;
            }
        }
        return minimax;
    }

    private double branchDoRemoveOrPass(double alpha, double beta) {
        PylosPlayerColor color = simulator.getColor();
        boolean isMaximizing = (color == PLAYER_COLOR);
        double minimax = isMaximizing ? INITIAL_THIS : INITIAL_OTHER;
        for (PylosSphere sphere : board.getSpheres(color)) {
            if (sphere.canRemove()) {
                PylosLocation prev = sphere.getLocation();
                simulator.removeSphere(sphere);
                double result = branchStep(alpha, beta);
                minimax = updateMinimax(minimax, result, isMaximizing);
                simulator.undoRemoveSecondSphere(sphere, prev, PylosGameState.REMOVE_SECOND, color);
                if (isMaximizing) alpha = Math.max(alpha, result); else beta = Math.min(beta, result);
                if (alpha >= beta) return minimax;
            }
        }
        simulator.pass();
        double result = branchStep(alpha, beta);
        simulator.undoPass(PylosGameState.REMOVE_SECOND, color);
        minimax = updateMinimax(minimax, result, isMaximizing);
        return minimax;
    }

    private double updateMinimax(double current, double candidate, boolean isMaximizing) {
        return isMaximizing ? Math.max(current, candidate) : Math.min(current, candidate);
    }

    /** --- MODIFIED METHOD --- */
    private double evaluatePosition() {
        long signature = computeBoardSignature();
        Double cachedScore = evaluationCache.get(signature);
        if (cachedScore != null) {
            return cachedScore;
        }

        int myReserves = board.getReservesSize(PLAYER_COLOR);
        int oppReserves = board.getReservesSize(PLAYER_COLOR.other());
        double score = (myReserves - oppReserves) * 100;
        score += evaluateHeightAdvantage() * 40;
        score += evaluateSquarePotential() * 30;
        score += (countAvailableMoves(PLAYER_COLOR) - countAvailableMoves(PLAYER_COLOR.other())) * 15;
        // Added trapped spheres evaluation
        score += evaluateTrappedSpheres() * 20;

        evaluationCache.put(signature, score);
        return score;
    }

    /** --- NEW METHOD --- */
    private double evaluateTrappedSpheres() {
        double score = 0;
        PylosSphere[] allSpheres = board.getSpheres();
        for (PylosSphere sphere : allSpheres) {
            if (sphere.isReserve()) continue;
            PylosLocation loc = sphere.getLocation();
            if (loc == null || loc.Z == 3) continue;

            int x = loc.X;
            int y = loc.Y;
            int z = loc.Z;
            boolean isTrapped = false;
            int[][] upperOffsets = {{0, 0}, {-1, 0}, {0, -1}, {-1, -1}};

            for (int[] offset : upperOffsets) {
                int upperX = x + offset[0];
                int upperY = y + offset[1];
                int upperZ = z + 1;
                if (upperX >= 0 && upperY >= 0 && upperX < (4 - upperZ) && upperY < (4 - upperZ)) {
                    PylosLocation upperLoc = board.getBoardLocation(upperX, upperY, upperZ);
                    if (upperLoc != null && upperLoc.getSphere() != null) {
                        isTrapped = true;
                        break;
                    }
                }
            }
            if (isTrapped) {
                if (sphere.PLAYER_COLOR == PLAYER_COLOR) score -= 15; else score += 15;
            }
        }
        return score;
    }

    private long computeBoardSignature() {
        long hash = 17;
        for (PylosLocation location : board.getLocations()) {
            hash = hash * 31 + encodeLocation(location);
        }
        hash = hash * 31 + simulator.getState().ordinal();
        hash = hash * 31 + simulator.getColor().ordinal();
        return hash;
    }

    private int encodeLocation(PylosLocation location) {
        PylosSphere sphere = location.getSphere();
        if (sphere == null) return 0;
        return sphere.PLAYER_COLOR == PLAYER_COLOR ? 1 : 2;
    }

    private double evaluateHeightAdvantage() {
        int[] myCounts = new int[4];
        int[] oppCounts = new int[4];
        countHeights(board.getSpheres(PLAYER_COLOR), myCounts);
        countHeights(board.getSpheres(PLAYER_COLOR.other()), oppCounts);
        double score = 0;
        for (int level = 0; level < 4; level++) {
            score += (myCounts[level] - oppCounts[level]) * Math.pow(2, level);
        }
        return score;
    }

    private void countHeights(PylosSphere[] spheres, int[] counts) {
        for (PylosSphere sphere : spheres) {
            if (!sphere.isReserve()) {
                PylosLocation loc = sphere.getLocation();
                if (loc != null) counts[loc.Z]++;
            }
        }
    }

    private double evaluateSquarePotential() {
        double score = 0;
        for (int level = 0; level < 3; level++) {
            int size = 4 - level;
            for (int x = 0; x < size - 1; x++) {
                for (int y = 0; y < size - 1; y++) {
                    score += evaluateSingleSquare(x, y, level);
                }
            }
        }
        return score;
    }

    private double evaluateSingleSquare(int x, int y, int level) {
        int myCount = 0, oppCount = 0;
        int[][] offsets = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
        for (int[] offset : offsets) {
            PylosLocation loc = board.getBoardLocation(x + offset[0], y + offset[1], level);
            if (loc != null && loc.getSphere() != null) {
                if (loc.getSphere().PLAYER_COLOR == PLAYER_COLOR) myCount++; else oppCount++;
            }
        }
        if (oppCount == 0) {
            if (myCount == 3) return 5.0;
            if (myCount == 2) return 2.0;
        }
        if (myCount == 0) {
            if (oppCount == 3) return -5.0;
            if (oppCount == 2) return -2.0;
        }
        return 0.0;
    }

    private int countAvailableMoves(PylosPlayerColor color) {
        int count = 0;
        PylosSphere[] spheres = board.getSpheres(color);
        PylosLocation[] locations = board.getLocations();
        for (PylosSphere sphere : spheres) {
            if (!sphere.isReserve()) {
                for (PylosLocation loc : locations) {
                    if (sphere.canMoveTo(loc)) count++;
                }
            }
        }
        if (board.getReservesSize(color) > 0) {
            for (PylosLocation loc : locations) {
                if (loc.isUsable()) count++;
            }
        }
        return count;
    }
}
