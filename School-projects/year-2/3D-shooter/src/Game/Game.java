package Game;

import javafx.animation.KeyFrame;
import javafx.animation.Timeline;
import javafx.application.Application;
import javafx.geometry.Bounds;
import javafx.scene.Scene;
import javafx.scene.image.Image;
import javafx.scene.paint.PhongMaterial;
import javafx.scene.shape.Box;
import javafx.stage.Stage;
import javafx.util.Duration;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Semaphore;

import Enemies.Enemy;
import Enemies.Spider;
import AnimatedObjects.Updatable;
import Enemies.Cultist;
import Item.HolyWater;
import Item.LargeMedKit;
import Item.PickUp;
import Item.RevolverAmmo;

/**
 * Static class handles running of application, initializing other static classes, map loading, game objects loading,
 * updating entities.
 */
public class Game extends Application
{
    /**
     * Size of square representing map tile in scene.
     */
    public static final int UNIT = 45;

    private static char[][] map;

    private static final Semaphore objectsMutex = new Semaphore(1);

    private static final Semaphore enemiesMutex = new Semaphore(1);

    private static final Semaphore pickUpsMutex = new Semaphore(1);

    private static List<Enemy> enemies = new ArrayList<>();

    private static List<Updatable> objects = new ArrayList<>();

    private static List<PickUp> pickUps = new ArrayList<>();

    private static boolean paused = false;

    private static int exitRow = 0, exitColumn = 0;


    /**
     * Launches game.
     * @param args
     */
    public static void main(String[] args)
    {
        launch(args);
    }

    /**
     * Returns the map represented as 2d array of character, where each character represents type of tile.
     * @return map.
     */
    public static char[][] getMap() { return map; }

    /**
     * @return whether the game is still running.
     */
    public static boolean isPaused() { return paused; }

    /**
     * Adds new object to list of update-able objects, mainly used for orbs.
     * @param object object to be regularly updated.
     */
    public static void addObject(Updatable object)
    {
        try
        {
            objectsMutex.acquire();
            objects.add(object);
            objectsMutex.release();
        }
        catch (InterruptedException ignored) { }
    }

    /**
     * Initializes application, plays loops.
     * @param primaryStage
     */
    @Override
    public void start(Stage primaryStage)
    {
        enemies = new ArrayList<>();
        objects = new ArrayList<>();
        paused = false;

        Scene scene = Renderer.init();
        GUI.init();

        readMapFile("src" + File.separator + "Assets" + File.separator + "levels" + File.separator + "level1.txt");

        primaryStage.setTitle("Dungeon of Horrors");
        primaryStage.setScene(scene);
        primaryStage.setMaximized(true);
        primaryStage.setResizable(false);
        primaryStage.toFront();
        primaryStage.show();

        EventsHandler.init(primaryStage);
        startEnemyLoop();
        startAnimatedObjectsLoop();
    }

    private static void startEnemyLoop()
    {
        List<Enemy> toBeRemoved = new ArrayList<>();
        Timeline enemyLoop = new Timeline(new KeyFrame(new Duration(50), event ->
        {
            if ( ! paused)
                for (Enemy enemy : enemies)
                    if (enemy.isInactive())
                        toBeRemoved.add(enemy);
                    else
                        enemy.update();
            try
            {
                enemiesMutex.acquire();
                enemies.removeAll(toBeRemoved);
                enemiesMutex.release();
            }
            catch (InterruptedException ignored) { }
        }));

        enemyLoop.setCycleCount(Timeline.INDEFINITE);
        enemyLoop.play();
    }

    private void startAnimatedObjectsLoop()
    {
        List<Updatable> toBeRemoved = new ArrayList<>();
        Timeline objectsLoop = new Timeline(new KeyFrame(new Duration(50), event ->
        {
            if ( ! paused)
                for (Updatable object : objects)
                    if (object.isInactive())
                        toBeRemoved.add(object);
                    else
                        object.update();
            try
            {
                objectsMutex.acquire();
                objects.removeAll(toBeRemoved);
                objectsMutex.release();
            }
            catch (InterruptedException ignored) { }
        }));

        objectsLoop.setCycleCount(Timeline.INDEFINITE);
        objectsLoop.play();
    }

    /**
     * Checks all spawned and active enemies and returns the first colliding with hitBox bounds.
     * @param hitBoxBounds entities hitBox.
     * @return colliding enemy or null if none.
     */
    public static Enemy collidingEnemy(Bounds hitBoxBounds)
    {
        try
        {
            enemiesMutex.acquire();
            for (Enemy enemy : enemies)
                if (enemy.getHIT_BOX().getBoundsInParent().intersects(hitBoxBounds))
                    return enemy;
        }
        catch (InterruptedException ignored) { }
        finally { enemiesMutex.release(); }

        return null;
    }

    /**
     * Checks all spawned items not picked up before and returns the first colliding with hitBox bounds.
     * @param hitBoxBounds entities hitBox.
     * @return colliding pick-up or null if none.
     */
    public static PickUp collidingPickUp(Bounds hitBoxBounds)
    {
        try
        {
            pickUpsMutex.acquire();
            pickUps.removeIf( i ->  ! i.isActive());
            for (PickUp pickUp : pickUps)
                if (pickUp.collision(hitBoxBounds))
                    return pickUp;

        }
        catch (InterruptedException ignored) { }
        finally { pickUpsMutex.release(); }

        return null;
    }

    static private void readMapFile(String file)
    {
        List<String> lines;
        try
        {
            lines = Files.readAllLines(Paths.get(file));
        }
        catch (IOException e)
        {
            e.printStackTrace();
            return;
        }

        int mapRows = lines.indexOf("---");
        map = new char[mapRows][];

        for (int i = 0; i < mapRows; i++)
        {
            String[] row =  lines.get(i).split(" ");
            map[i] = new char[row.length];

            for (int j = 0; j < row.length; j++)
            {
                map[i][j] = row[j].charAt(0);
            }
        }
        loadMap();

        int lineIndex = mapRows + 1;

        for (; lineIndex < lines.size(); lineIndex++)
        {
            String[] row = lines.get(lineIndex).split(" ");
            String type = row[0];
            double x = Integer.parseInt(row[1]) * UNIT + (UNIT / 2d);
            double z = Integer.parseInt(row[2]) * UNIT + (UNIT / 2d);

            switch (type)
            {
                case "P" -> Player.init(x, z);
                case "E" -> initExit(Integer.parseInt(row[2]), Integer.parseInt(row[1]));
                case "S" -> enemies.add(new Spider(x,0, z));
                case "C" -> enemies.add(new Cultist(x,0, z));
                case "A" -> pickUps.add(new RevolverAmmo(x, 0, z));
                case "H" -> pickUps.add(new HolyWater(x, 0, z));
                case "M" -> pickUps.add(new LargeMedKit(x, 0, z));
            }
        }
    }

    private static void initExit(int row, int column)
    {
        exitRow = row;
        exitColumn = column;
    }

    /**
     * Checks if entities coordinates are occupying game exit tile.
     * @param x x coordinate of entity.
     * @param z z coordinate of entity.
     * @return whether exit was reached.
     */
    public static boolean exitReached(double x, double z)
    {
        int row = (int)Math.floor(z / UNIT);
        int column = (int)Math.floor(x / UNIT);
        return exitRow == row && exitColumn == column;
    }

    /**
     * Stops game and initializes "you win" screen.
     */
    public static void win()
    {
        paused = true;
        GUI.initWinScreen();
    }

    /**
     * Stops game and initializes "you lose" screen.
     * @param deathByInsanity Whether player lost all their sanity or health.
     */
    public static void lose(boolean deathByInsanity)
    {
        paused = true;
        GUI.initGameOverScreen(deathByInsanity);
    }

    static private void loadMap()
    {
        for (int i = 0; i < map.length; i++)
        {
            int Z = i * UNIT;
            for (int j = 0; j < map[0].length; j++)
            {
                int X = j * 45;
                if (map[i][j] == 'S')
                    createBox(X, Z);
                if (map[i][j] == '.' || map[i][j] == 'D')
                    createFloor(X, Z);
                if (map[i][j] == 'E')
                    createGate(X, Z);
            }
        }
    }

    static private void createBox(int X, int Z)
    {
        Box box = new Box(UNIT, UNIT, UNIT);
        box.setTranslateX(X + (UNIT / 2.0));
        box.setTranslateZ(Z + (UNIT / 2.0));
        box.setTranslateY(UNIT / -2.0);

        PhongMaterial mat = new PhongMaterial();
        mat.setDiffuseMap(new Image("Assets" + File.separator + "textures" + File.separator + "stoneWall.png"));
        box.setMaterial(mat);

        Renderer.getGroup().getChildren().add(box);
    }

    static private void createGate(int X, int Z)
    {
        Box box = new Box(UNIT, UNIT, UNIT);
        box.setTranslateX(X + (UNIT / 2.0));
        box.setTranslateZ(Z + (UNIT / 2.0));
        box.setTranslateY(UNIT / -2.0);

        PhongMaterial mat = new PhongMaterial();
        mat.setDiffuseMap(new Image("Assets" + File.separator + "textures" + File.separator + "exit.png"));
        box.setMaterial(mat);

        Renderer.getGroup().getChildren().add(box);
    }

    static private void createFloor(int X, int Z)
    {
        Box box = new Box(UNIT, UNIT, UNIT);
        box.setTranslateX(X + (UNIT / 2.0));
        box.setTranslateZ(Z + (UNIT / 2.0));
        box.setTranslateY(UNIT / 2.0);

        PhongMaterial mat = new PhongMaterial();
        mat.setDiffuseMap(new Image("Assets" + File.separator + "textures" + File.separator + "stoneFloor.png"));
        box.setMaterial(mat);

        Renderer.getGroup().getChildren().add(box);
    }

    /**
     * For debugging purposes. Prints the map and marks the tile occupied by player by x.
     */
    static public void printMap()
    {
        int column = (int)Math.floor(Player.getX() / Game.UNIT);
        int row = (int)Math.floor(Player.getZ() / Game.UNIT);
        char[][] map = Game.getMap();

        for (int i = 0; i < map.length; i++)
        {
            for (int j = 0; j < map[i].length; j++)
            {
                if (i == row && j == column)
                    System.out.print("x");
                else
                    System.out.print(map[i][j]);
            }
            System.out.println();
        }
        System.out.println("column: " + column);
        System.out.println("row: " + row);
    }
}
