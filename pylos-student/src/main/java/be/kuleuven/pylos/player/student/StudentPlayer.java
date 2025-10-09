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
         * 	PylosSphere[] mySpheres = board.getSpheres(this);
         * 	PylosSphere myReserveSphere = board.getReserve(this); */

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
    }

    @Override
    public void doRemove(PylosGameIF game, PylosBoard board) {
        /* game methods
         * game.removeSphere(mySphere); */
    }

    @Override
    public void doRemoveOrPass(PylosGameIF game, PylosBoard board) {
        /* game methods
         * game.removeSphere(mySphere);
         * game.pass() */
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
        if (square_possible()) weight_by_move += RESERVE_WEIGHT;

        // you get an advantage by having a higher
        return weight_by_move;
    }
    private boolean square_possible(PylosBoard board , PylosLocation location) {
        int[][][] offsets = {
                {{0,0},{1,0},{0,1},{1,1}},
                {{-1,0},{0,0},{-1,1},{0,1}},
                {{0,-1},{1,-1},{0,0},{1,0}},
                {{-1,-1},{0,-1},{-1,0},{0,0}}
        };
        PylosSphere[] mySpheres = board.getSpheres(this);
        for (PylosSphere sphere : mySpheres) {
            if (sphere.getLocation().Z == location.Z
                    &&( sphere.getLocation().X == location.X -1 &&  sphere.getLocation().Y == location.Y
                    ||  sphere.getLocation().X == location.X +1 &&  sphere.getLocation().Y == location.Y
                    ||  sphere.getLocation().X == location.X  &&  sphere.getLocation().Y == location.Y -1
                    ||  sphere.getLocation().X == location.X  &&  sphere.getLocation().Y == location.Y +2
                    ))  {return true;}

        }
        return false;
    }


}
