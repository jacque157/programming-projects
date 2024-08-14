using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TransformFloors : MonoBehaviour
{
    public TorchPuzzle puzzleController;
    public GameObject firstFloor;
    public GameObject secondFloor;
    public GameObject thirdFloor;
    public GameObject StartHall;
    public GameObject EndHall;
    public GameObject EntryHall;

    public Vector3 firstFloorPosition;
    public Vector3 secondFloorPosition;
    public Vector3 thirdFloorPosition;


    // Start is called before the first frame update
    void Start()
    {
        firstFloorPosition = firstFloor.transform.position;
        secondFloorPosition = secondFloor.transform.position;
        thirdFloorPosition = thirdFloor.transform.position;
    }

    // Update is called once per frame
    void Update()
    {
            
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.tag == "Player" && this.enabled)
        {
            puzzleController.currentLevel += 1;
            firstFloor.transform.position = firstFloorPosition - ((puzzleController.currentLevel) * new Vector3(0, 3f * 5f, 0));
            secondFloor.transform.position = secondFloorPosition - ((puzzleController.currentLevel) * new Vector3(0, 3f * 5f, 0));
            thirdFloor.transform.position = thirdFloorPosition - ((puzzleController.currentLevel) * new Vector3(0, 3f * 5f, 0));

            if (puzzleController.IsSolved())
            {
                EndHall.SetActive(true);
                EntryHall.SetActive(false);

                firstFloor.SetActive(false);
                secondFloor.SetActive(false);
                thirdFloor.SetActive(false);

                puzzleController.DisableDistantTorches();
            }
            else
            {
                puzzleController.ResetPuzzle();
                puzzleController.ResetTorchesPosiotions();
                EndHall.SetActive(false);
                EntryHall.SetActive(true);
            }

            

            StartHall.SetActive(true);

            

            enabled = false; // Stupid wayaround for stupid unity
            //gameObject.SetActive(false); 
        }
    }
}
