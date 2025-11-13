package be.kuleuven.pylos.player.student;

import be.kuleuven.pylos.game.*;
import be.kuleuven.pylos.player.PylosPlayer;

import java.util.HashMap;

/**
 * Created by Jan on 20/02/2015.
 */
public class StudentCodex extends PylosPlayer {
    private PylosGameSimulator simulator;
    private PylosBoard board;
    final private int maxBranchDepth = 5;
    private double bestMinimax;
    private PylosSphere bestSphere;
    private PylosLocation bestLocation;
    private final double INITIAL_THIS = -9999;

    private HashMap<Long, Double> minimaxResults;

    public double evaluateBoard(PylosBoard board, PylosPlayer player) {
        PylosPlayerColor myColor = player.PLAYER_COLOR;
        PylosPlayerColor oppColor = myColor.other();

        PylosSphere[] mySpheres = board.getSpheres(myColor);
        PylosSphere[] oppSpheres = board.getSpheres(oppColor);

        int myReserve = board.getReservesSize(myColor);
        int oppReserve = board.getReservesSize(oppColor);

        int myOnBoard = 0;
        int oppOnBoard = 0;
        int myHeightSum = 0;
        int oppHeightSum = 0;
        int myMobile = 0;
        int oppMobile = 0;
        double myCenterControl = 0;
        double oppCenterControl = 0;

        for (PylosSphere sphere : mySpheres) {
            if (sphere.isReserve()) {
                continue;
            }
            PylosLocation location = sphere.getLocation();
            myOnBoard++;
            myHeightSum += location.Z;
            if (sphere.canMove()) {
                myMobile++;
            }
            myCenterControl += centerBonus(location);
        }

        for (PylosSphere sphere : oppSpheres) {
            if (sphere.isReserve()) {
                continue;
            }
            PylosLocation location = sphere.getLocation();
            oppOnBoard++;
            oppHeightSum += location.Z;
            if (sphere.canMove()) {
                oppMobile++;
            }
            oppCenterControl += centerBonus(location);
        }

        int myFullSquares = 0;
        int oppFullSquares = 0;
        int myAlmostSquares = 0;
        int oppAlmostSquares = 0;
        int myPotentialSquares = 0;
        int oppPotentialSquares = 0;
        int myTopControl = 0;
        int oppTopControl = 0;

        for (PylosSquare square : board.getAllSquares()) {
            int mine = square.getInSquare(myColor);
            int opp = square.getInSquare(oppColor);

            if (mine == 4) {
                myFullSquares++;
            }
            if (opp == 4) {
                oppFullSquares++;
            }

            if (opp == 0) {
                if (mine == 3) {
                    myAlmostSquares++;
                } else if (mine == 2) {
                    myPotentialSquares++;
                }
            }
            if (mine == 0) {
                if (opp == 3) {
                    oppAlmostSquares++;
                } else if (opp == 2) {
                    oppPotentialSquares++;
                }
            }

            PylosLocation top = square.getTopLocation();
            if (top != null && !top.isUsed()) {
                if (mine == 4) {
                    myTopControl++;
                } else if (opp == 4) {
                    oppTopControl++;
                }
            }
        }

        double score = 0;
        score += (myReserve - oppReserve) * 160; // ++
        score += (myMobile - oppMobile) * 12; // ++
        score += (myFullSquares - oppFullSquares) * 220; // +++
        score += (myAlmostSquares - oppAlmostSquares) * 90; // +++
        score += (myPotentialSquares - oppPotentialSquares) * 28;
        score += (myTopControl - oppTopControl) * 60;
        score += (myCenterControl - oppCenterControl) * 80; // ++

        return score;
    }

    private double centerBonus(PylosLocation location) {
        int size = 4 - location.Z;
        double centerX = (size - 1) / 2.0;
        double centerY = (size - 1) / 2.0;
        double dx = Math.abs(location.X - centerX);
        double dy = Math.abs(location.Y - centerY);
        double distance = dx + dy;
        double maxDistance = centerX + centerY;
        double normalized = maxDistance == 0 ? 0 : (maxDistance - distance) / maxDistance;
        return normalized * (location.Z + 1);
    }

    public int min_max(int currentDepth, boolean maxPlayer) {
        PylosPlayerColor winner = simulator.getWinner();
        if (winner != null) {
            if (winner == PLAYER_COLOR) {
                return 2000 - currentDepth;
            }
            return -2000 + currentDepth;
        }

        if (currentDepth >= maxBranchDepth) {
            return (int) Math.round(evaluateBoard(board, this));
        }

        long cacheKey = computeCacheKey(currentDepth, maxPlayer);
        Double cached = minimaxResults.get(cacheKey);
        if (cached != null) {
            return cached.intValue();
        }

        double bestValue = maxPlayer ? Double.NEGATIVE_INFINITY : Double.POSITIVE_INFINITY;
        boolean explored = false;

        PylosGameState state = simulator.getState();
        PylosPlayerColor currentColor = simulator.getColor();
        PylosLocation[] allLocations = board.getLocations();
        PylosSphere[] spheres = board.getSpheres(currentColor);

        switch (state) {
            case MOVE:
                if (board.getReservesSize(currentColor) > 0) {
                    PylosSphere reserveSphere = board.getReserve(currentColor);
                    for (PylosLocation location : allLocations) {
                        if (!location.isUsable()) {
                            continue;
                        }
                        explored = true;
                        PylosGameState prevState = simulator.getState();
                        PylosPlayerColor prevColor = simulator.getColor();
                        simulator.moveSphere(reserveSphere, location);
                        boolean nextMax = simulator.getColor() == PLAYER_COLOR;
                        double score = min_max(currentDepth + 1, nextMax);
                        simulator.undoAddSphere(reserveSphere, prevState, prevColor);
                        bestValue = updateBest(bestValue, score, maxPlayer);
                    }
                }
                for (PylosSphere sphere : spheres) {
                    if (sphere.isReserve() || !sphere.canMove()) {
                        continue;
                    }
                    PylosLocation fromLocation = sphere.getLocation();
                    for (PylosLocation location : allLocations) {
                        if (!sphere.canMoveTo(location)) {
                            continue;
                        }
                        explored = true;
                        PylosGameState prevState = simulator.getState();
                        PylosPlayerColor prevColor = simulator.getColor();
                        simulator.moveSphere(sphere, location);
                        boolean nextMax = simulator.getColor() == PLAYER_COLOR;
                        double score = min_max(currentDepth + 1, nextMax);
                        simulator.undoMoveSphere(sphere, fromLocation, prevState, prevColor);
                        bestValue = updateBest(bestValue, score, maxPlayer);
                    }
                }
                break;
            case REMOVE_FIRST:
                for (PylosSphere sphere : spheres) {
                    if (!sphere.canRemove()) {
                        continue;
                    }
                    explored = true;
                    PylosLocation previousLocation = sphere.getLocation();
                    PylosGameState prevState = simulator.getState();
                    PylosPlayerColor prevColor = simulator.getColor();
                    simulator.removeSphere(sphere);
                    boolean nextMax = simulator.getColor() == PLAYER_COLOR;
                    double score = min_max(currentDepth + 1, nextMax);
                    simulator.undoRemoveFirstSphere(sphere, previousLocation, prevState, prevColor);
                    bestValue = updateBest(bestValue, score, maxPlayer);
                }
                break;
            case REMOVE_SECOND:
                for (PylosSphere sphere : spheres) {
                    if (!sphere.canRemove()) {
                        continue;
                    }
                    explored = true;
                    PylosLocation previousLocation = sphere.getLocation();
                    PylosGameState prevState = simulator.getState();
                    PylosPlayerColor prevColor = simulator.getColor();
                    simulator.removeSphere(sphere);
                    boolean nextMax = simulator.getColor() == PLAYER_COLOR;
                    double score = min_max(currentDepth + 1, nextMax);
                    simulator.undoRemoveSecondSphere(sphere, previousLocation, prevState, prevColor);
                    bestValue = updateBest(bestValue, score, maxPlayer);
                }
                explored = true;
                PylosGameState prevState = simulator.getState();
                PylosPlayerColor prevColor = simulator.getColor();
                simulator.pass();
                boolean nextMax = simulator.getColor() == PLAYER_COLOR;
                double passScore = min_max(currentDepth + 1, nextMax);
                simulator.undoPass(prevState, prevColor);
                bestValue = updateBest(bestValue, passScore, maxPlayer);
                break;
            default:
                break;
        }

        if (!explored) {
            bestValue = evaluateBoard(board, this);
        }

        minimaxResults.put(cacheKey, bestValue);
        return (int) Math.round(bestValue);
    }

    private long computeCacheKey(int currentDepth, boolean maxPlayer) {
        long key = board.toLong();
        key = key * 31 + simulator.getState().ordinal();
        key = key * 31 + simulator.getColor().ordinal();
        key = key * 31 + (maxPlayer ? 1 : 0);
        key = key * 31 + currentDepth;
        return key;
    }

    private double updateBest(double currentBest, double candidate, boolean maxPlayer) {
        return maxPlayer ? Math.max(currentBest, candidate) : Math.min(currentBest, candidate);
    }

    private void init(PylosGameState state, PylosBoard board) {
        //noinspection DuplicatedCode
        this.simulator = new PylosGameSimulator(state, PLAYER_COLOR, board);
        this.board = board;
        this.bestMinimax = INITIAL_THIS;
        this.bestSphere = null;
        this.bestLocation = null;
        this.minimaxResults = new HashMap<>();
    }

    @Override
    public void doMove(PylosGameIF game, PylosBoard board) {
        init(game.getState(), board);
        PylosLocation[] allLocations = board.getLocations();
        PylosSphere[] mySpheres = board.getSpheres(this);
        PylosSphere myReserveSphere = board.getReservesSize(this) > 0 ? board.getReserve(this) : null;

        // game methods
        if (board.getSpheres().length == 0) {
            PylosLocation start = board.getBoardLocation(1, 1, 0);
            if (myReserveSphere != null) {
                game.moveSphere(myReserveSphere, start);
            }
            return;
        }

        if (myReserveSphere != null) {
            for (PylosLocation location : allLocations) {
                if (!location.isUsable()) {
                    continue;
                }
                PylosGameState prevState = simulator.getState();
                PylosPlayerColor prevColor = simulator.getColor();
                simulator.moveSphere(myReserveSphere, location);
                boolean nextMax = simulator.getColor() == PLAYER_COLOR;
                double score = min_max(1, nextMax);
                simulator.undoAddSphere(myReserveSphere, prevState, prevColor);
                if (score > bestMinimax) {
                    bestMinimax = score;
                    bestSphere = myReserveSphere;
                    bestLocation = location;
                }
            }
        }

        for (PylosSphere sphere : mySpheres) {
            if (sphere.isReserve() || !sphere.canMove()) {
                continue;
            }
            PylosLocation fromLocation = sphere.getLocation();
            for (PylosLocation location : allLocations) {
                if (!sphere.canMoveTo(location)) {
                    continue;
                }
                PylosGameState prevState = simulator.getState();
                PylosPlayerColor prevColor = simulator.getColor();
                simulator.moveSphere(sphere, location);
                boolean nextMax = simulator.getColor() == PLAYER_COLOR;
                double score = min_max(1, nextMax);
                simulator.undoMoveSphere(sphere, fromLocation, prevState, prevColor);
                if (score > bestMinimax) {
                    bestMinimax = score;
                    bestSphere = sphere;
                    bestLocation = location;
                }
            }
        }

        if (bestSphere != null && bestLocation != null) {
            game.moveSphere(bestSphere, bestLocation);
            return;
        }

        if (myReserveSphere != null) {
            for (PylosLocation location : allLocations) {
                if (location.isUsable()) {
                    game.moveSphere(myReserveSphere, location);
                    return;
                }
            }
        }

        for (PylosSphere sphere : mySpheres) {
            if (sphere.isReserve() || !sphere.canMove()) {
                continue;
            }
            for (PylosLocation location : allLocations) {
                if (sphere.canMoveTo(location)) {
                    game.moveSphere(sphere, location);
                    return;
                }
            }
        }
    }

    @Override
    public void doRemove(PylosGameIF game, PylosBoard board) {
        /* game methods
         * game.removeSphere(mySphere); */
        init(game.getState(), board);

        for (PylosSphere sphere : board.getSpheres(this)) {
            if (!sphere.canRemove()) {
                continue;
            }
            PylosLocation previousLocation = sphere.getLocation();
            PylosGameState prevState = simulator.getState();
            PylosPlayerColor prevColor = simulator.getColor();
            simulator.removeSphere(sphere);
            boolean nextMax = simulator.getColor() == PLAYER_COLOR;
            double score = min_max(1, nextMax);
            simulator.undoRemoveFirstSphere(sphere, previousLocation, prevState, prevColor);
            if (score > bestMinimax) {
                bestMinimax = score;
                bestSphere = sphere;
                bestLocation = null;
            }
        }

        if (bestSphere == null) {
            for (PylosSphere sphere : board.getSpheres(this)) {
                if (sphere.canRemove()) {
                    bestSphere = sphere;
                    break;
                }
            }
        }

        if (bestSphere != null) {
            game.removeSphere(bestSphere);
        }
    }

    @Override
    public void doRemoveOrPass(PylosGameIF game, PylosBoard board) {
        /* game methods
         * game.removeSphere(mySphere);
         * game.pass() */
        init(game.getState(), board);

        for (PylosSphere sphere : board.getSpheres(this)) {
            if (!sphere.canRemove()) {
                continue;
            }
            PylosLocation previousLocation = sphere.getLocation();
            PylosGameState prevState = simulator.getState();
            PylosPlayerColor prevColor = simulator.getColor();
            simulator.removeSphere(sphere);
            boolean nextMax = simulator.getColor() == PLAYER_COLOR;
            double score = min_max(1, nextMax);
            simulator.undoRemoveSecondSphere(sphere, previousLocation, prevState, prevColor);
            if (score > bestMinimax) {
                bestMinimax = score;
                bestSphere = sphere;
                bestLocation = null;
            }
        }

        PylosGameState prevState = simulator.getState();
        PylosPlayerColor prevColor = simulator.getColor();
        simulator.pass();
        boolean nextMax = simulator.getColor() == PLAYER_COLOR;
        double passScore = min_max(1, nextMax);
        simulator.undoPass(prevState, prevColor);
        if (passScore > bestMinimax) {
            bestMinimax = passScore;
            bestSphere = null;
            bestLocation = null;
        }

        if (bestSphere != null) {
            game.removeSphere(bestSphere);
        } else {
            game.pass();
        }
    }
}