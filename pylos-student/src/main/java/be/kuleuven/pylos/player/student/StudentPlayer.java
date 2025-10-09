package be.kuleuven.pylos.player.student;

import be.kuleuven.pylos.game.*;
import be.kuleuven.pylos.player.PylosPlayer;

import java.util.HashMap;

public class StudentPlayer extends PylosPlayer {
    private PylosGameSimulator simulator;
    private PylosBoard board;
    private double bestMinimax;
    private PylosSphere bestSphere;
    private PylosLocation bestLocation;
    private int branchDepth;
    private HashMap<Long, Double> minimaxResults;

    private static final double WIN_THIS = 2000;
    private static final double WIN_OTHER = -2000;
    private static final double INITIAL_THIS = -9999;
    private static final double INITIAL_OTHER = 9999;
    private static final int MAX_BRANCH_DEPTH = 4;

    @Override
    public void doMove(PylosGameIF game, PylosBoard board) {
        init(game.getState(), board);

        PylosSphere myReserveSphere = board.getReserve(this);
        PylosSphere[] mySpheres = board.getSpheres(this);
        PylosLocation[] locations = board.getLocations();

        // Try moving existing spheres to higher levels
        if (myReserveSphere != null) {
            for (PylosSphere sphere : mySpheres) {
                if (!sphere.isReserve()) {
                    for (PylosLocation location : locations) {
                        if (sphere.canMoveTo(location)) {
                            tryMove(sphere, location);
                        }
                    }
                }
            }
        }

        // Try adding reserve sphere
        for (PylosLocation location : locations) {
            if (location.isUsable()) {
                tryAddReserve(myReserveSphere, location);
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

        // Try passing
        simulator.pass();
        eval(branchStep(bestMinimax, bestMinimax), null, null);
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
        this.minimaxResults = new HashMap<>();
    }

    private void eval(double minimax, PylosSphere sphere, PylosLocation location) {
        if (minimax > bestMinimax) {
            bestMinimax = minimax;
            bestSphere = sphere;
            bestLocation = location;
        }
    }

    // Helper methods to reduce code duplication
    private void tryMove(PylosSphere sphere, PylosLocation location) {
        PylosLocation prevLocation = sphere.getLocation();
        simulator.moveSphere(sphere, location);
        eval(branchStep(bestMinimax, bestMinimax), sphere, location);
        simulator.undoMoveSphere(sphere, prevLocation, PylosGameState.MOVE, PLAYER_COLOR);
    }

    private void tryAddReserve(PylosSphere sphere, PylosLocation location) {
        simulator.moveSphere(sphere, location);
        eval(branchStep(bestMinimax, bestMinimax), sphere, location);
        simulator.undoAddSphere(sphere, PylosGameState.MOVE, PLAYER_COLOR);
    }

    private void tryRemove(PylosSphere sphere, PylosGameState state) {
        PylosLocation prevLocation = sphere.getLocation();
        simulator.removeSphere(sphere);
        eval(branchStep(bestMinimax, bestMinimax), sphere, null);

        if (state == PylosGameState.REMOVE_FIRST) {
            simulator.undoRemoveFirstSphere(sphere, prevLocation, state, PLAYER_COLOR);
        } else {
            simulator.undoRemoveSecondSphere(sphere, prevLocation, state, PLAYER_COLOR);
        }
    }

    private double branchStep(double siblingMinimax, double parentSiblingMinimax) {
        if (branchDepth == MAX_BRANCH_DEPTH) {
            return evaluatePosition();
        }

        PylosPlayerColor color = simulator.getColor();
        PylosGameState state = simulator.getState();

        // Check cache
        long hash = computeBoardHash(color, state);
        Double cached = minimaxResults.get(hash);
        if (cached != null) return cached;

        branchDepth++;
        double result;

        switch (state) {
            case MOVE:
                result = branchDoMove(siblingMinimax);
                break;
            case REMOVE_FIRST:
                result = branchDoRemove(parentSiblingMinimax);
                break;
            case REMOVE_SECOND:
                result = branchDoRemoveOrPass(parentSiblingMinimax);
                break;
            case COMPLETED:
                result = (simulator.getWinner() == PLAYER_COLOR ? WIN_THIS : WIN_OTHER);
                result += (simulator.getWinner() == PLAYER_COLOR ? -branchDepth : branchDepth);
                break;
            case DRAW:
                result = simulator.getWinner() == PLAYER_COLOR ? WIN_OTHER : WIN_THIS;
                break;
            default:
                throw new IllegalStateException("Game state is: " + state);
        }

        branchDepth--;
        minimaxResults.put(hash, result);
        return result;
    }

    private double branchDoMove(double siblingMinimax) {
        PylosPlayerColor color = simulator.getColor();
        boolean isMaximizing = (color == PLAYER_COLOR);
        double minimax = isMaximizing ? INITIAL_THIS : INITIAL_OTHER;

        PylosSphere reserve = board.getReserve(color);
        PylosSphere[] spheres = board.getSpheres(color);
        PylosLocation[] locations = board.getLocations();

        // Try moving spheres to higher levels
        if (reserve != null) {
            for (PylosSphere sphere : spheres) {
                if (!sphere.isReserve()) {
                    for (PylosLocation loc : locations) {
                        if (sphere.canMoveTo(loc)) {
                            PylosLocation prev = sphere.getLocation();
                            simulator.moveSphere(sphere, loc);
                            double result = branchStep(minimax, siblingMinimax);
                            minimax = updateMinimax(minimax, result, isMaximizing);
                            simulator.undoMoveSphere(sphere, prev, PylosGameState.MOVE, color);
                        }
                    }
                }
            }
        }

        // Try adding reserve sphere
        for (PylosLocation loc : locations) {
            if (loc.isUsable()) {
                simulator.moveSphere(reserve, loc);
                double result = branchStep(minimax, siblingMinimax);
                minimax = updateMinimax(minimax, result, isMaximizing);
                simulator.undoAddSphere(reserve, PylosGameState.MOVE, color);
            }
        }

        return minimax;
    }

    private double branchDoRemove(double parentSiblingMinimax) {
        PylosPlayerColor color = simulator.getColor();
        boolean isMaximizing = (color == PLAYER_COLOR);
        double minimax = isMaximizing ? INITIAL_THIS : INITIAL_OTHER;

        for (PylosSphere sphere : board.getSpheres(color)) {
            if (sphere.canRemove()) {
                PylosLocation prev = sphere.getLocation();
                simulator.removeSphere(sphere);
                double result = branchStep(parentSiblingMinimax, parentSiblingMinimax);
                minimax = updateMinimax(minimax, result, isMaximizing);
                simulator.undoRemoveFirstSphere(sphere, prev, PylosGameState.REMOVE_FIRST, color);
            }
        }

        return minimax;
    }

    private double branchDoRemoveOrPass(double parentSiblingMinimax) {
        PylosPlayerColor color = simulator.getColor();
        boolean isMaximizing = (color == PLAYER_COLOR);
        double minimax = isMaximizing ? INITIAL_THIS : INITIAL_OTHER;

        for (PylosSphere sphere : board.getSpheres(color)) {
            if (sphere.canRemove()) {
                PylosLocation prev = sphere.getLocation();
                simulator.removeSphere(sphere);
                double result = branchStep(minimax, minimax);
                minimax = updateMinimax(minimax, result, isMaximizing);
                simulator.undoRemoveSecondSphere(sphere, prev, PylosGameState.REMOVE_SECOND, color);
            }
        }

        // Pass option
        simulator.pass();
        double result = evaluatePosition();
        minimax = updateMinimax(minimax, result, isMaximizing);
        simulator.undoPass(PylosGameState.REMOVE_SECOND, color);

        return minimax;
    }

    // Consolidate minimax update logic
    private double updateMinimax(double current, double candidate, boolean isMaximizing) {
        return isMaximizing ? Math.max(current, candidate) : Math.min(current, candidate);
    }

    private long computeBoardHash(PylosPlayerColor currentColor, PylosGameState state) {
        long hash = 0;
        PylosLocation[] locations = board.getLocations();

        for (int i = 0; i < locations.length; i++) {
            PylosLocation loc = locations[i];
            if (loc.getSphere() != null) {
                int colorValue = loc.getSphere().PLAYER_COLOR == PLAYER_COLOR ? 1 : 2;
                hash = hash * 31 + colorValue * (i + 1);
            }
        }

        hash = hash * 31 + board.getReservesSize(PLAYER_COLOR);
        hash = hash * 31 + board.getReservesSize(PLAYER_COLOR.other());
        hash = hash * 31 + (currentColor == PLAYER_COLOR ? 1 : 2);
        hash = hash * 31 + state.ordinal();

        return hash;
    }

    private double evaluatePosition() {
        int myReserves = board.getReservesSize(PLAYER_COLOR);
        int oppReserves = board.getReservesSize(PLAYER_COLOR.other());

        double score = (myReserves - oppReserves) * 100;
        score += evaluateHeightAdvantage() * 50;
        score += evaluateSquarePotential() * 30;
        score += evaluateMobility() * 10;

        return score;
    }

    private double evaluateHeightAdvantage() {
        int[] myCounts = new int[4];
        int[] oppCounts = new int[4];

        countHeights(board.getSpheres(PLAYER_COLOR), myCounts);
        countHeights(board.getSpheres(PLAYER_COLOR.other()), oppCounts);

        double score = 0;
        for (int level = 0; level < 4; level++) {
            int weight = 1 << level; // 2^level
            score += (myCounts[level] - oppCounts[level]) * weight;
        }

        return score;
    }

    private void countHeights(PylosSphere[] spheres, int[] counts) {
        for (PylosSphere sphere : spheres) {
            if (!sphere.isReserve()) {
                PylosLocation loc = sphere.getLocation();
                if (loc != null) {
                    counts[loc.Z]++;
                }
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
        int myCount = 0, oppCount = 0, emptyCount = 0;
        int[][] offsets = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};

        for (int[] offset : offsets) {
            PylosLocation loc = board.getBoardLocation(x + offset[0], y + offset[1], level);
            PylosSphere sphere = loc.getSphere();

            if (sphere == null) {
                emptyCount++;
            } else if (sphere.PLAYER_COLOR == PLAYER_COLOR) {
                myCount++;
            } else {
                oppCount++;
            }
        }

        // Only evaluate unmixed squares
        if (oppCount == 0 && myCount == 3) return 5.0;
        if (oppCount == 0 && myCount == 2) return 2.0;
        if (myCount == 0 && oppCount == 3) return -5.0;
        if (myCount == 0 && oppCount == 2) return -2.0;

        return 0.0;
    }

    private double evaluateMobility() {
        return countAvailableMoves(PLAYER_COLOR) - countAvailableMoves(PLAYER_COLOR.other());
    }

    private int countAvailableMoves(PylosPlayerColor color) {
        int count = 0;
        PylosSphere[] spheres = board.getSpheres(color);
        PylosLocation[] locations = board.getLocations();

        // Count moves to higher levels
        for (PylosSphere sphere : spheres) {
            if (!sphere.isReserve()) {
                for (PylosLocation loc : locations) {
                    if (sphere.canMoveTo(loc)) {
                        count++;
                    }
                }
            }
        }

        // Count reserve placements
        if (board.getReservesSize(color) > 0) {
            for (PylosLocation loc : locations) {
                if (loc.isUsable()) {
                    count++;
                }
            }
        }

        return count;
    }
}