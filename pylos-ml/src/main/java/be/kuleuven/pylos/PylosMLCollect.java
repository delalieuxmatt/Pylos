package be.kuleuven.pylos;

import be.kuleuven.pylos.battle.BattleMT;
import be.kuleuven.pylos.battle.BattleResult;
import be.kuleuven.pylos.battle.data.PlayedGame;
import be.kuleuven.pylos.player.PylosPlayer;
import be.kuleuven.pylos.player.PylosPlayerType;
import be.kuleuven.pylos.player.codes.PylosPlayerBestFit;
import be.kuleuven.pylos.player.codes.PylosPlayerMiniMax;
import com.google.gson.*;
import org.tensorflow.SavedModelBundle;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.util.List;


public class PylosMLCollect {
    public static final String EXPORT_PATH = "pylos-ml/src/main/training/resources/games/0.json";
    public final static String MODEL_PATH = "pylos-ml/src/main/training/resources/models/latest";

    public static void main(String[] args) throws IOException {
        // Collect games
        List<PlayedGame> playedGames = PylosMLCollect.collectGames();

        System.out.println("Played games: " + playedGames.size());

        // Export to json file
        File file = new File(EXPORT_PATH);
        Files.createDirectories(file.getParentFile().toPath());
        FileWriter writer = new FileWriter(file);
        Gson gson = new GsonBuilder().create();

        gson.toJson(playedGames, writer);

        writer.flush();
        writer.close();

        System.out.println("Exported to: " + EXPORT_PATH);
    }

    public static List<PlayedGame> collectGames() {
        SavedModelBundle model = SavedModelBundle.load(MODEL_PATH, "serve");
        PylosPlayerType p1 = new PylosPlayerType("PML_1") {
            @Override
            public PylosPlayer create() {
                return new PylosPlayerML(model);
            }
        };
        PylosPlayerType p2 = new PylosPlayerType("PML_2") {
            @Override
            public PylosPlayer create() {
                return new PylosPlayerML(model);
            }
        };

        BattleResult br = BattleMT.play(p1, p2, 10000, 8, true);

        return br.playedGames;
    }

//    public static List<PlayedGame> collectGames() {
//        SavedModelBundle model = SavedModelBundle.load(MODEL_PATH, "serve");
//        PylosPlayerType p1 = new PylosPlayerType("MM_1") {
//            @Override
//            public PylosPlayer create() {
//                return new PylosPlayerMiniMax(3);
//            }
//        };
//        PylosPlayerType p2 = new PylosPlayerType("MM3_2") {
//            @Override
//            public PylosPlayer create() {
//                return new PylosPlayerMiniMax(3);
//            }
//        };

//        BattleResult br = BattleMT.play(p1, p2, 10000, 8, true);
//
//        return br.playedGames;
//    }
}


