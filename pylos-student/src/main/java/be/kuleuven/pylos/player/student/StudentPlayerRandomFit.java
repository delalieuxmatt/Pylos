package be.kuleuven.pylos.player.student;

import be.kuleuven.pylos.game.PylosBoard;
import be.kuleuven.pylos.game.PylosGameIF;
import be.kuleuven.pylos.game.PylosLocation;
import be.kuleuven.pylos.game.PylosSphere;
import be.kuleuven.pylos.player.PylosPlayer;

import java.util.ArrayList;
import java.util.Random;

/**
 * Created by Ine on 5/05/2015.
 */
public class StudentPlayerRandomFit extends PylosPlayer {
    Random rand = new Random();


    @Override
    public void doMove(PylosGameIF game, PylosBoard board) {
        PylosLocation[] allLocations = board.getLocations();
        ArrayList<PylosLocation> availableLocations = new ArrayList<>();
        for(int i = 0; i<allLocations.length; i++){
            if (allLocations[i].isUsable()) {
                availableLocations.add(allLocations[i]);
            }
        }
        int randomInt = rand.nextInt(availableLocations.size());
        PylosSphere myReserveSphere = board.getReserve(this);
        game.moveSphere(myReserveSphere, availableLocations.get(randomInt));


        /* add a reserve sphere to a feasible random location */

    }

    @Override
    public void doRemove(PylosGameIF game, PylosBoard board) {
        PylosSphere[] mySpheres = board.getSpheres(this);
        int randomInt = getRandom().nextInt(mySpheres.length);
        game.removeSphere(mySpheres[randomInt]);
    }

    @Override
    public void doRemoveOrPass(PylosGameIF game, PylosBoard board) {
        game.pass();

    }
}
