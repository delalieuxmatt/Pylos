package be.kuleuven.pylos.player.student;

import be.kuleuven.pylos.game.*;
import be.kuleuven.pylos.player.PylosPlayer;

import java.util.ArrayList;
import java.util.HashMap;

public class StudentPlayer extends PylosPlayer {
    private PylosLocation remove1 = null;
    private PylosLocation remove2 = null;
    private PylosGameSimulator simulator;
    private PylosBoard board;
    private double bestMinimax;
    private PylosSphere bestSphere;
    private PylosLocation bestLocation;
    private int branchDepth = 0;
    private HashMap<Long, Double> minimaxResults;

    private final double WIN_THIS = 2000;
    private final double WIN_OTHER = -2000;
    private final double INITIAL_THIS = -9999;
    private final double INITIAL_OTHER = 9999;
    private int MAX_BRANCH_DEPTH = 4;

    @Override
    public void doMove(PylosGameIF game, PylosBoard board) {
        init(game.getState(), board);

        PylosSphere myReserveSphere = board.getReserve(this);
        PylosSphere[] mySpheres = board.getSpheres(this);
        PylosLocation[] locations = board.getLocations();

        // Try to move a sphere to higher level
        if (myReserveSphere != null) {
            for (PylosSphere sphere : mySpheres) {
                if (!sphere.isReserve()) {
                    for (PylosLocation location : locations) {
                        if (sphere.canMoveTo(location)) {
                            PylosLocation prevLocation = sphere.getLocation();
                            simulator.moveSphere(sphere, location);
                            double minimax = branchStep(bestMinimax, bestMinimax);
                            eval(minimax, sphere, location);
                            simulator.undoMoveSphere(sphere, prevLocation, PylosGameState.MOVE, this.PLAYER_COLOR);
                        }
                    }
                }
            }
        }

        // Try to add a reserve sphere
        for (PylosLocation location : locations) {
            if (location.isUsable()) {
                simulator.moveSphere(myReserveSphere, location);
                double minimax = branchStep(bestMinimax, bestMinimax);
                eval(minimax, myReserveSphere, location);
                simulator.undoAddSphere(myReserveSphere, PylosGameState.MOVE, this.PLAYER_COLOR);
            }
        }

        // Execute the best move
        game.moveSphere(bestSphere, bestLocation);
    }

    @Override
    public void doRemove(PylosGameIF game, PylosBoard board) {
        init(game.getState(), board);

        for (PylosSphere sphere : board.getSpheres(PLAYER_COLOR)) {
            if (sphere.canRemove()) {
                PylosLocation prevLocation = sphere.getLocation();
                simulator.removeSphere(sphere);
                double minimax = branchStep(bestMinimax, bestMinimax);
                eval(minimax, sphere, null);
                simulator.undoRemoveFirstSphere(sphere, prevLocation, PylosGameState.REMOVE_FIRST, this.PLAYER_COLOR);
            }
        }

        // Execute the best move
        game.removeSphere(bestSphere);
    }

    @Override
    public void doRemoveOrPass(PylosGameIF game, PylosBoard board) {
        init(game.getState(), board);

        for (PylosSphere sphere : board.getSpheres(PLAYER_COLOR)) {
            if (sphere.canRemove()) {
                PylosLocation prevLocation = sphere.getLocation();
                simulator.removeSphere(sphere);
                double minimax = branchStep(bestMinimax, bestMinimax);
                eval(minimax, sphere, null);
                simulator.undoRemoveSecondSphere(sphere, prevLocation, PylosGameState.REMOVE_SECOND, this.PLAYER_COLOR);
            }
        }

        // Try passing
        simulator.pass();
        double chance = branchStep(bestMinimax, bestMinimax);
        eval(chance, null, null);
        simulator.undoPass(PylosGameState.REMOVE_SECOND, this.PLAYER_COLOR);

        // Execute the best move
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

    /**
     * Main branching method that delegates to specific game state handlers
     */
    private double branchStep(double siblingMinimax, double parentSiblingMinimax) {
        // Base case: max depth reached
        if (branchDepth == MAX_BRANCH_DEPTH) {
            return evaluatePosition(board);
        }

        final PylosPlayerColor color = simulator.getColor();
        final PylosGameState state = simulator.getState();

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
                result = simulator.getWinner() == PLAYER_COLOR ? WIN_THIS : WIN_OTHER;
                if (simulator.getWinner() == PLAYER_COLOR) {
                    result -= branchDepth;
                } else {
                    result += branchDepth;
                }
                break;
            case DRAW:
                result = simulator.getWinner() == PLAYER_COLOR ? WIN_OTHER : WIN_THIS;
                break;
            default:
                throw new IllegalStateException("Game state is: " + state);
        }

        branchDepth--;
        return result;
    }

    /**
     * Handles the MOVE state - tries all possible moves
     */
    private double branchDoMove(double siblingMinimax) {
        final PylosPlayerColor currentColor = simulator.getColor();
        double minimax = currentColor == PLAYER_COLOR ? INITIAL_THIS : INITIAL_OTHER;

        PylosSphere myReserveSphere = board.getReserve(currentColor);
        PylosSphere[] mySpheres = board.getSpheres(currentColor);
        PylosLocation[] locations = board.getLocations();

        // Try to move a sphere to higher level
        if (myReserveSphere != null) {
            for (PylosSphere sphere : mySpheres) {
                if (!sphere.isReserve()) {
                    for (PylosLocation location : locations) {
                        if (sphere.canMoveTo(location)) {
                            PylosLocation prevLocation = sphere.getLocation();
                            simulator.moveSphere(sphere, location);
                            double result = branchStep(minimax, siblingMinimax);

                            if (currentColor == PLAYER_COLOR) {
                                if (result > minimax) minimax = result;
                            } else {
                                if (result < minimax) minimax = result;
                            }

                            simulator.undoMoveSphere(sphere, prevLocation, PylosGameState.MOVE, currentColor);
                        }
                    }
                }
            }
        }

        // Try to add a reserve sphere
        for (PylosLocation location : locations) {
            if (location.isUsable()) {
                simulator.moveSphere(myReserveSphere, location);
                double result = branchStep(minimax, siblingMinimax);

                if (currentColor == PLAYER_COLOR) {
                    if (result > minimax) minimax = result;
                } else {
                    if (result < minimax) minimax = result;
                }

                simulator.undoAddSphere(myReserveSphere, PylosGameState.MOVE, currentColor);
            }
        }

        return minimax;
    }

    /**
     * Handles the REMOVE_FIRST state - must remove a sphere after completing a square
     */
    private double branchDoRemove(double parentSiblingMinimax) {
        final PylosPlayerColor currentColor = simulator.getColor();
        double minimax = currentColor == PLAYER_COLOR ? INITIAL_THIS : INITIAL_OTHER;

        PylosSphere[] mySpheres = board.getSpheres(currentColor);

        // Remove a sphere
        for (PylosSphere sphere : mySpheres) {
            if (sphere.canRemove()) {
                PylosLocation prevLocation = sphere.getLocation();
                simulator.removeSphere(sphere);
                double result = branchStep(parentSiblingMinimax, parentSiblingMinimax);

                if (currentColor == PLAYER_COLOR) {
                    if (result > minimax) minimax = result;
                } else {
                    if (result < minimax) minimax = result;
                }

                simulator.undoRemoveFirstSphere(sphere, prevLocation, PylosGameState.REMOVE_FIRST, currentColor);
            }
        }

        return minimax;
    }

    /**
     * Handles the REMOVE_SECOND state - can remove a second sphere or pass
     */
    private double branchDoRemoveOrPass(double parentSiblingMinimax) {
        final PylosPlayerColor currentColor = simulator.getColor();
        double minimax = currentColor == PLAYER_COLOR ? INITIAL_THIS : INITIAL_OTHER;

        PylosSphere[] mySpheres = board.getSpheres(currentColor);

        // Remove a sphere
        for (PylosSphere sphere : mySpheres) {
            if (sphere.canRemove()) {
                PylosLocation prevLocation = sphere.getLocation();
                simulator.removeSphere(sphere);
                double result = branchStep(minimax, minimax);

                if (currentColor == PLAYER_COLOR) {
                    if (result > minimax) minimax = result;
                } else {
                    if (result < minimax) minimax = result;
                }

                simulator.undoRemoveSecondSphere(sphere, prevLocation, PylosGameState.REMOVE_SECOND, currentColor);
            }
        }

        // Pass
        simulator.pass();
        double result = evaluatePosition(board);

        if (currentColor == PLAYER_COLOR) {
            if (result > minimax) minimax = result;
        } else {
            if (result < minimax) minimax = result;
        }

        simulator.undoPass(PylosGameState.REMOVE_SECOND, currentColor);

        return minimax;
    }

    /**
     * Enhanced evaluation function considering multiple strategic factors
     */
    private double evaluatePosition(PylosBoard board) {
        double score = 0;

        // 1. RESERVE DIFFERENCE (most important - running out of reserves loses the game)
        int myReserves = board.getReservesSize(PLAYER_COLOR);
        int oppReserves = board.getReservesSize(PLAYER_COLOR.other());
        score += (myReserves - oppReserves) * 100;  // Weight: 100

        // 2. HEIGHT ADVANTAGE (spheres on higher levels are more valuable)
        score += evaluateHeightAdvantage(board) * 50;  // Weight: 50

        // 3. SQUARE POTENTIAL (ability to form squares for sphere removal)
        score += evaluateSquarePotential(board) * 30;  // Weight: 30

        // 4. MOBILITY (number of available moves)
        score += evaluateMobility(board) * 10;  // Weight: 10

        return score;
    }

    /**
     * Evaluates advantage based on sphere heights
     * Higher level spheres are more valuable as they're closer to winning
     */
    private double evaluateHeightAdvantage(PylosBoard board) {
        double heightScore = 0;
        PylosSphere[] mySpheres = board.getSpheres(PLAYER_COLOR);
        PylosSphere[] oppSpheres = board.getSpheres(PLAYER_COLOR.other());

        // Count spheres at each level with increasing weights
        int[] myHeightCounts = new int[4];
        int[] oppHeightCounts = new int[4];

        for (PylosSphere sphere : mySpheres) {
            if (!sphere.isReserve()) {
                PylosLocation loc = sphere.getLocation();
                if (loc != null) {
                    myHeightCounts[loc.Z]++;
                }
            }
        }

        for (PylosSphere sphere : oppSpheres) {
            if (!sphere.isReserve()) {
                PylosLocation loc = sphere.getLocation();
                if (loc != null) {
                    oppHeightCounts[loc.Z]++;
                }
            }
        }

        // Weight spheres by height: level 0=1, level 1=2, level 2=4, level 3=8
        for (int level = 0; level < 4; level++) {
            int weight = (int) Math.pow(2, level);
            heightScore += (myHeightCounts[level] - oppHeightCounts[level]) * weight;
        }

        return heightScore;
    }

    /**
     * Evaluates potential to form squares (for removing spheres)
     */
    private double evaluateSquarePotential(PylosBoard board) {
        double squareScore = 0;

        // Check all possible square positions on each level
        for (int level = 0; level < 3; level++) {  // Levels 0-2 can have squares
            int boardSize = 4 - level;

            for (int x = 0; x < boardSize - 1; x++) {
                for (int y = 0; y < boardSize - 1; y++) {
                    // Check 2x2 square starting at (x, y, level)
                    squareScore += evaluateSingleSquare(board, x, y, level);
                }
            }
        }

        return squareScore;
    }

    /**
     * Evaluates a single 2x2 square position
     */
    private double evaluateSingleSquare(PylosBoard board, int x, int y, int level) {
        int myCount = 0;
        int oppCount = 0;
        int emptyCount = 0;

        // Check all 4 positions of the square
        int[][] offsets = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};

        for (int[] offset : offsets) {
            PylosLocation loc = board.getBoardLocation(x + offset[0], y + offset[1], level);

            if (loc.getSphere() == null) {
                emptyCount++;
            } else if (loc.getSphere().PLAYER_COLOR == PLAYER_COLOR) {
                myCount++;
            } else {
                oppCount++;
            }
        }

        // Score potential squares
        if (oppCount == 0) {  // Only my spheres or empty
            if (myCount == 3 && emptyCount == 1) {
                return 5.0;  // One move away from completing square!
            } else if (myCount == 2 && emptyCount == 2) {
                return 2.0;  // Partial square potential
            }
        } else if (myCount == 0) {  // Only opponent spheres or empty
            if (oppCount == 3 && emptyCount == 1) {
                return -5.0;  // Opponent one move away!
            } else if (oppCount == 2 && emptyCount == 2) {
                return -2.0;  // Opponent has partial square
            }
        }

        return 0.0;  // Mixed square - no potential
    }

    /**
     * Evaluates mobility (number of available moves)
     * More options = better position
     */
    private double evaluateMobility(PylosBoard board) {
        int myMoves = countAvailableMoves(board, PLAYER_COLOR);
        int oppMoves = countAvailableMoves(board, PLAYER_COLOR.other());

        return myMoves - oppMoves;
    }

    /**
     * Counts available moves for a player
     */
    /**
     * Counts available moves for a player
     */
    /**
     * Counts available moves for a player
     */
    private int countAvailableMoves(PylosBoard board, PylosPlayerColor color) {
        int moveCount = 0;
        PylosSphere[] spheres = board.getSpheres(color);
        PylosLocation[] locations = board.getLocations();

        // 1. Count moves to higher levels
        for (PylosSphere sphere : spheres) {
            if (!sphere.isReserve()) {
                for (PylosLocation loc : locations) {
                    if (sphere.canMoveTo(loc)) {
                        moveCount++;
                    }
                }
            }
        }

        // 2. Count reserve placement moves
        // *** NEW FIX: Check the reserve size (which is guaranteed not to crash) ***
        int reserveSize = board.getReservesSize(color);

        if (reserveSize > 0) { // Only proceed if a reserve sphere actually exists
            // Since we know one exists, we can calculate the moves
            // (We no longer need to call PylosSphere reserve = board.getReserve(color);
            // which was causing the crash)

            for (PylosLocation loc : locations) {
                if (loc.isUsable()) {
                    moveCount++;
                }
            }
        }

        return moveCount;
    }


}
