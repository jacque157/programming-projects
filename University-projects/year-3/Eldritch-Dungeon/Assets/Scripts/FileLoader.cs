using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FileLoader : MonoBehaviour
{
    public TextAsset levelGeometry;
    public TextAsset levelObjects;

    public GameObject revolverAmmoPrefab;
    public GameObject bandagePrefab;
    public GameObject medKitPrefab;
    public GameObject blueHerbPrefab;
    public GameObject holyWaterPrefab;

    public GameObject spiderPrefab;
    public GameObject cultistPrefab;
    public GameObject wallPrefab;
    public GameObject floorPrefab;
    public GameObject doorPrefab;
    public GameObject exitPrefab;

    // Start is called before the first frame update
    void Start()
    {
        float z = 0;
        float y = Map.scale / 2;

        foreach (string line in levelGeometry.ToString().Split('\n'))
        {
            float x = 0;
            foreach (string type in line.Split(' '))
            {
                CreateObject(x, y, z, type.Trim());
                x += Map.scale;
            }
            z += Map.scale;

            List<string> row = new List<string>(line.Split());
            Map.map.Add(row);
        }
        Map.loaded = true;

        foreach (string line in levelObjects.ToString().Split('\n'))
        {
            string[] elements = line.Trim().Split(' ');
            float x = float.Parse(elements[1]) * Map.scale;
            z = float.Parse(elements[2]) * Map.scale;
            CreateObject(x, y, z, elements[0]);
        }
    }

    public void CreateObject(float x, float y, float z, string type)
    {
        switch (type)
        {
            case "S":
                CreateStoneWall(x, y, z);
                break;
            case ".":
                CreateStoneFloor(x, y, z);
                break;
            case "P":
                CreatePlayer(x, 0, z);
                break;
            case "Sp":
                CreateSpider(x, 0, z);
                break;
            case "C":
                CreateCultist(x, 0, z);
                break;
            case "AR":
                CreateRevolverAmmo(x, 0, z);
                break;
            case "MS":
                CreateBandage(x, 0, z);
                break;
            case "HS":
                CreateBlueHerb(x, 0, z);
                break;
            case "ML":
                CreateHealthKit(x, 0, z);
                break;
            case "HL":
                CreateHolyWater(x, 0, z);
                break;
            case "E":
                CreateExit(x, y, z);
                break;
            case "D":
                CreateDoor(x, y, z);
                break;
            default:
                break;             
        }
    }

    private void CreateStoneWall(float x, float y, float z)
    {
        GameObject wall = Instantiate(wallPrefab, new Vector3(x, y, z), Quaternion.identity);
        wall.transform.localScale = new Vector3(Map.scale, Map.scale, Map.scale);
    }

    private void CreateStoneFloor(float x, float y, float z)
    {
        GameObject floor = Instantiate(floorPrefab, new Vector3(x, y - Map.scale, z), Quaternion.identity);
        floor.transform.localScale = new Vector3(Map.scale, Map.scale, Map.scale);
    }

    private void CreateExit(float x, float y, float z)
    {
        GameObject exit = Instantiate(exitPrefab, new Vector3(x, y, z), Quaternion.identity);
        exit.transform.localScale = new Vector3(Map.scale, Map.scale, Map.scale);
        Map.exitTrigger = exit.GetComponent<BoxCollider>();
    }

    private void CreateDoor(float x, float y, float z)
    {
        GameObject door = Instantiate(doorPrefab, new Vector3(x, y, z), Quaternion.identity);
        door.transform.localScale = new Vector3(Map.scale, Map.scale, Map.scale);
    }

    private void CreatePlayer(float x, float y, float z)
    {
        GameObject player = GameObject.Find("Player");
        float y1 = y;
        player.transform.position = new Vector3(x, y1, z);
    }

    private void CreateSpider(float x, float y, float z)
    {
        float scale = 0.5f;
        GameObject spider = Instantiate(spiderPrefab, new Vector3(x, y, z), Quaternion.identity);
        spider.transform.localScale = new Vector3(scale, scale, scale);
        float y1 = (scale / 2);
        spider.transform.position = new Vector3(x, y1, z);
    }

    private void CreateCultist(float x, float y, float z)
    {
        float scaleX = 0.75f;
        float scaleY = 0.7f;
        float scaleZ = 0.75f;

        float groundOffset = 0.9f;
        GameObject cultist = Instantiate(cultistPrefab, new Vector3(x, y, z), Quaternion.identity);
        cultist.transform.localScale = new Vector3(scaleX, scaleY, scaleZ);
        cultist.transform.position = new Vector3(x, groundOffset, z);
    }

    private void CreateBandage(float x, float y, float z)
    {
        float scale = 0.4f;
        GameObject bandage = Instantiate(bandagePrefab, new Vector3(x, y, z), Quaternion.identity);
        bandage.transform.localScale = new Vector3(scale, scale, scale);
        float y1 = 0.214f;
        bandage.transform.position = new Vector3(x, y1, z);
    }

    private void CreateHealthKit(float x, float y, float z)
    {
        float scale = 0.4f;
        GameObject kit = Instantiate(medKitPrefab, new Vector3(x, y, z), Quaternion.identity);
        kit.transform.localScale = new Vector3(scale, scale, scale);
        float y1 = 0.32f;
        kit.transform.position = new Vector3(x, y1, z);
    }

    private void CreateHolyWater(float x, float y, float z)
    {
        float scale = 0.4f;
        GameObject water = Instantiate(holyWaterPrefab, new Vector3(x, y, z), Quaternion.identity);
        water.transform.localScale = new Vector3(scale, scale, scale);
        float y1 = 0.427f;
        water.transform.position = new Vector3(x, y1, z);
    }

    private void CreateBlueHerb(float x, float y, float z)
    {
        float scale = 0.75f;
        GameObject herb = Instantiate(blueHerbPrefab, new Vector3(x, y, z), Quaternion.identity);
        herb.transform.localScale = new Vector3(scale, scale, scale);
        float y1 = 0.153f;
        herb.transform.position = new Vector3(x, y1, z);
    }

    private void CreateRevolverAmmo(float x, float y, float z)
    {
        float scale = 1f;
        GameObject ammo = Instantiate(revolverAmmoPrefab, new Vector3(x, y, z), Quaternion.identity);
        ammo.transform.localScale = new Vector3(scale, scale, scale);   
        float y1 = 0.231f;
        ammo.transform.position = new Vector3(x, y1, z);
    }
}
