using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.InputSystem.EnhancedTouch;
using UnityEngine.UIElements;

public class TorchPuzzle : MonoBehaviour
{
    public List<GameObject> firstFloorTorches = new List<GameObject>();
    public GameObject firstFloorCorrectTorch;
    public List<GameObject> secondFloorTorches = new List<GameObject>();
    public GameObject secondFloorCorrectTorch;
    public List<GameObject> thirdFloorTorches = new List<GameObject>();
    public GameObject thirdFloorCorrectTorch;

    public GameObject torch1;
    public GameObject torch2;
    public float maxTorchDistance = 2.0f;
    public GameObject player;

    public int currentLevel = 0;

    GameObject firstFloorLit;
    GameObject secondFloorLit;
    GameObject thirdFloorLit;

    //Vector3 torch1StartPosition;
    //Vector3 torch2StartPosition;

    Vector3 torch1RelativeStartPosition;
    Quaternion torch1startRotation;
    Vector3 torch2RelativeStartPosition;
    Quaternion torch2startRotation;

    // Start is called before the first frame update
    void Start()
    {
        torch1RelativeStartPosition = torch1.transform.position - transform.position;
        torch2RelativeStartPosition = torch2.transform.position - transform.position;
        torch1startRotation = torch1.transform.rotation;
        torch2startRotation = torch2.transform.rotation;
        //torch1StartPosition = torch1.transform.position;
        //print(torch1StartPosition);
        //torch2StartPosition = torch2.transform.position;
        //print(torch2StartPosition);
    }

    // Update is called once per frame
    void Update()
    {
        updateTorches();
    }

    public void ResetTorchesPosiotions()
    {
        float distance1 = (player.transform.position - torch1.transform.position).magnitude;
        if (distance1 > maxTorchDistance) 
        {
            torch1.transform.position = transform.position + torch1RelativeStartPosition - (currentLevel * new Vector3(0, 3f * 5f, 0)); ;
            torch1.transform.rotation = torch1startRotation;
        }
        float distance2 = (player.transform.position - torch2.transform.position).magnitude;
        if (distance2 > maxTorchDistance)
        {
            torch2.transform.position = transform.position + torch2RelativeStartPosition - (currentLevel * new Vector3(0, 3f * 5f, 0)); ;
            torch2.transform.rotation = torch2startRotation;
        }
    }

    public void DisableDistantTorches()
    {
        float distance1 = (player.transform.position - torch1.transform.position).magnitude;
        if (distance1 > maxTorchDistance)
        {
            torch1.SetActive(false);
        }
        float distance2 = (player.transform.position - torch2.transform.position).magnitude;
        if (distance2 > maxTorchDistance)
        {
            torch2.SetActive(false);
        }
    }

    void updateTorches()
    {
        foreach (GameObject torch in firstFloorTorches)
        {
            if (FlameIsActive(torch.gameObject.GetComponent<Flame>()))
            {
                if (firstFloorLit == null)
                {
                    firstFloorLit = torch;
                }
                if (torch != firstFloorLit)
                {
                    ResetFirstFloorTorches();
                    firstFloorLit = torch;
                    ActivateFlame(torch.GetComponent<Flame>());
                }
            }
        }

        foreach (GameObject torch in secondFloorTorches)
        {
            if (FlameIsActive(torch.gameObject.GetComponent<Flame>()))
            {
                if (secondFloorLit == null)
                {
                    secondFloorLit = torch;
                }
                if (torch != secondFloorLit)
                {
                    ResetSecondFloorTorches();
                    secondFloorLit = torch;
                    ActivateFlame(torch.GetComponent<Flame>());
                }
            }
        }

        foreach (GameObject torch in thirdFloorTorches)
        {
            if (FlameIsActive(torch.gameObject.GetComponent<Flame>()))
            {
                if (thirdFloorLit == null)
                {
                    thirdFloorLit = torch;
                }
                if (torch != thirdFloorLit)
                {
                    ResetThirdFloorTorches();
                    thirdFloorLit = torch;
                    ActivateFlame(torch.GetComponent<Flame>());
                }
            }
        }
    }

    public void ResetPuzzle()
    {
        firstFloorLit = secondFloorLit = thirdFloorLit = null;
        ResetFirstFloorTorches();
        ResetSecondFloorTorches();
        ResetThirdFloorTorches();
    }

    public bool IsSolved()
    { 
        return firstFloorLit == firstFloorCorrectTorch && secondFloorLit == secondFloorCorrectTorch && thirdFloorLit == thirdFloorCorrectTorch;
    }
    
    void ActivateFlame(Flame flame)
    {
        //flame.flameEffects.SetActive(true);
        flame.flameEffects.SetActive(true);
        flame.onFire = true;
    }

    void DisableFlame(Flame flame)
    {
        //flame.flameEffects.SetActive(false);
        flame.flameEffects.SetActive(false);
        flame.onFire = false;
    }

    bool FlameIsActive(Flame flame)
    {
        return flame.onFire;
    }

    void ResetFirstFloorTorches() 
    {
        foreach(GameObject torch in firstFloorTorches) 
        {
            DisableFlame(torch.gameObject.GetComponent<Flame>());
        }
    }

    void ResetSecondFloorTorches()
    {
        foreach (GameObject torch in secondFloorTorches)
        {
            DisableFlame(torch.gameObject.GetComponent<Flame>());
        }
    }

    void ResetThirdFloorTorches()
    {
        foreach (GameObject torch in thirdFloorTorches)
        {
            DisableFlame(torch.gameObject.GetComponent<Flame>());
        }
    }

}
