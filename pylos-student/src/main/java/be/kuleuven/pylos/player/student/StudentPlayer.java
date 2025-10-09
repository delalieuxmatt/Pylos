package be.kuleuven.pylos.player.student;

import be.kuleuven.pylos.game.PylosBoard;
import be.kuleuven.pylos.game.PylosGameIF;
import be.kuleuven.pylos.game.PylosLocation;
import be.kuleuven.pylos.game.PylosSphere;
import be.kuleuven.pylos.player.PylosPlayer;

/**
 * Created by Jan on 20/02/2015.
 */
public class StudentPlayer extends PylosPlayer {
    private PylosLocation remove1 = null;
    private PylosLocation remove2 = null;
    @Override
    public void doMove(PylosGameIF game, PylosBoard board) {
        /* board methods
         * 	PylosLocation[] allLocations = board.getLocations();
         * 	PylosSphere[] allSpheres = board.getSpheres();
         * 	PylosSphere[] mySpheres = board.getSpheres(this); */
         PylosSphere myReserveSphere = board.getReserve(this);

        /* game methods
         * game.moveSphere(myReserveSphere, allLocations[0]); */
        double currentbestmove = Integer.MIN_VALUE;
        PylosLocation currentbestlocation = null;
        PylosLocation[] allLocations = board.getLocations();
        PylosSphere[] allSpheres = board.getSpheres();
        PylosSphere[] mySpheres = board.getSpheres(this); // gets availlable spheres
        for (PylosLocation loc : board.getLocations()) {
            if (loc.isUsable()) {
                double value = evaluate_Move(game,board, loc);
                if (value >  currentbestmove) {
                    currentbestlocation = loc;
                    currentbestmove = value;
                }
            }
        }
        if (currentbestlocation != null && myReserveSphere != null) {
            game.moveSphere(myReserveSphere, currentbestlocation);
        }
    }

    @Override
    public void doRemove(PylosGameIF game, PylosBoard board) {
        // Remove the lowest sphere that doesn't support anything
        PylosSphere[] mySpheres = board.getSpheres(this);

        for (int z = 0; z < 4; z++) {  // Start from lowest level
            for (PylosSphere sphere : mySpheres) {
                PylosLocation loc = sphere.getLocation();
                if (loc != null && loc.Z == z && !sphere.isReserve()) {
                    if (canRemoveSphere(sphere, board)) {
                        game.removeSphere(sphere);
                        return;
                    }
                }
            }
        }
    }

    @Override
    public void doRemoveOrPass(PylosGameIF game, PylosBoard board) {
        // Try to remove a sphere, otherwise pass
        PylosSphere[] mySpheres = board.getSpheres(this);

        for (int z = 0; z < 4; z++) {  // Start from lowest level
            for (PylosSphere sphere : mySpheres) {
                PylosLocation loc = sphere.getLocation();
                if (loc != null && loc.Z == z && !sphere.isReserve()) {
                    if (canRemoveSphere(sphere, board)) {
                        game.removeSphere(sphere);
                        return;
                    }
                }
            }
        }

        // If no sphere can be removed, pass
        game.pass();
    }

    /**
     * Checks if a sphere can be removed (i.e., it's not supporting any other spheres)
     */
    private boolean canRemoveSphere(PylosSphere sphere, PylosBoard board) {
        PylosLocation loc = sphere.getLocation();
        if (loc == null) {
            return false;
        }

        // A sphere at position (X, Y, Z) supports 4 possible positions at level Z+1:
        // (X, Y, Z+1), (X-1, Y, Z+1), (X, Y-1, Z+1), (X-1, Y-1, Z+1)
        // If any of these positions have a sphere, this sphere cannot be removed

        PylosSphere[] allSpheres = board.getSpheres();
        int nextLevel = loc.Z + 1;

        // If we're already at the top level, nothing can be above us
        if (nextLevel >= 4) {
            return true;
        }

        for (PylosSphere otherSphere : allSpheres) {
            PylosLocation otherLoc = otherSphere.getLocation();
            if (otherLoc != null && otherLoc.Z == nextLevel) {
                // Check if otherSphere is supported by our sphere
                // A sphere at (X', Y', Z+1) is supported by spheres at:
                // (X', Y', Z), (X'+1, Y', Z), (X', Y'+1, Z), (X'+1, Y'+1, Z)
                int dx = otherLoc.X - loc.X;
                int dy = otherLoc.Y - loc.Y;

                if ((dx == 0 || dx == 1) && (dy == 0 || dy == 1)) {
                    return false; // This sphere supports something above
                }
            }
        }

        return true; // No spheres above, safe to remove
    }


    /**
     *
     * this function needs to return a double on good this move is , we calculate our chance after this move and the
     * chance that the other player wins
     *  then we subtract our chance with the other player's chance and this gives a result
     *  the higher the result the better the option.
     *  Do not calculate the current probability , this has no change of outcome in the game
     */

    private double evaluate_Move(PylosGameIF game ,PylosBoard board, PylosLocation location) {
        double weight_by_move = 0;
        final double step_weight = 1.0;
        final double HEIGHT_WEIGHT = 4.0; // * height
        final double SQUARE_WEIGHT = 25.0; // making a square leads to having a bigger reserve and higher chance to win
        final double RESERVE_WEIGHT = 8.0; // this does not matter in this calc , but its an indication


        weight_by_move += HEIGHT_WEIGHT * location.Z + 1; // plus one is so that there is a feasibel solution
        if (square_possible(board, location)) weight_by_move += RESERVE_WEIGHT;

        // you get an advantage by having a higher
        return weight_by_move;
    }
    private boolean square_possible(PylosBoard board, PylosLocation location) {
        // A location can be part of up to 4 different 2x2 squares on its level
        // Check each possible square where this location is one of the 4 corners

        int[][] squareOffsets = {
                // Each row: [x1, y1, x2, y2, x3, y3, x4, y4] for the 4 corners
                // Square where location is TOP-LEFT corner
                {0, 0, 1, 0, 0, 1, 1, 1},
                // Square where location is TOP-RIGHT corner
                {-1, 0, 0, 0, -1, 1, 0, 1},
                // Square where location is BOTTOM-LEFT corner
                {0, -1, 1, -1, 0, 0, 1, 0},
                // Square where location is BOTTOM-RIGHT corner
                {-1, -1, 0, -1, -1, 0, 0, 0}
        };

        PylosSphere[] mySpheres = board.getSpheres(this);
        int boardSize = 4 - location.Z; // Board dimensions for this level

        for (int[] offsets : squareOffsets) {
            int mySpheresInSquare = 0;
            boolean validSquare = true;

            // Check all 4 positions of this potential square
            for (int i = 0; i < 4; i++) {
                int x = location.X + offsets[i * 2];
                int y = location.Y + offsets[i * 2 + 1];

                // Check if position is within bounds
                if (x < 0 || y < 0 || x >= boardSize || y >= boardSize) {
                    validSquare = false;
                    break;
                }

                // Skip the location where we're about to place (it will become ours)
                if (x == location.X && y == location.Y) {
                    continue;
                }

                // Check if any of my spheres occupies this position
                boolean hasMySphere = false;
                for (PylosSphere sphere : mySpheres) {
                    PylosLocation sphereLoc = sphere.getLocation();
                    if (sphereLoc != null && !sphere.isReserve() &&
                            sphereLoc.Z == location.Z &&
                            sphereLoc.X == x && sphereLoc.Y == y) {
                        hasMySphere = true;
                        mySpheresInSquare++;
                        break;
                    }
                }

                // If this position doesn't have my sphere, this square won't work
                if (!hasMySphere) {
                    validSquare = false;
                    break;
                }
            }

            // If we found a valid square with 3 of our spheres already placed,
            // placing at 'location' will complete the square
            if (validSquare && mySpheresInSquare == 3) {
                return true;
            }
        }

        return false;
    }



}
