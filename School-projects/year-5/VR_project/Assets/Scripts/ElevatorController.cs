using System.Collections;
using System.Collections.Generic;
using Unity.XR.CoreUtils;
using UnityEngine;
using UnityEngine.XR.Interaction.Toolkit;

public class ElevatorController : MonoBehaviour
{
    public Animator mAnimator;
    private bool isOpen = false;
    public GameObject player;
    public GameObject teleportLocation;
    public XRDirectInteractor LHinteractor;
    public XRDirectInteractor RHinteractor;

    public bool enableDebug = false;
    public ElevatorController mirrorElevator;
    public bool tempDisabled = false;
    private int playerLayer;
    private void Start()
    {
        playerLayer = LayerMask.NameToLayer ( "Player" );
    }


    private void OnTriggerEnter(Collider other)
    {
        if(other.transform.gameObject.layer !=  playerLayer)
            return;
        if (enableDebug)
            Debug.Log("IN");
        if (mAnimator != null && !isOpen)
        {
            mAnimator.SetTrigger("TrOpen");
            isOpen = true;
            //also open the mirorElevator
            mirrorElevator.mAnimator.SetTrigger("TrOpen");
            mirrorElevator.isOpen = true;
        }
    }

    private void OnTriggerExit(Collider other)
    {
        if(other.transform.gameObject.layer !=  playerLayer)
            return;
        //exited wrong way
        if(player.transform.position.x < this.gameObject.transform.position.x)
            return;
        if (tempDisabled) 
            StartCoroutine(ReenableSelfTeleport());

        if (enableDebug)
            Debug.Log("Out");
        if (mAnimator != null && isOpen)
        {
            mAnimator.SetTrigger("TrClose");
            isOpen = false;
            //player.transform.position = teleportLocation.transform.position;
            // but dont close the mirror
            // mirrorElevator.mAnimator.SetTrigger("TrClose");
            // mirrorElevator.isOpen = false;

            //allow animator but dont teleport
            if (tempDisabled) return;

            StartTeleport();
        }
    }

    public void StartTeleport()
    {
        var xrOrigin = player.GetComponent<XROrigin>();
        var newPos = teleportLocation.transform.position;
        newPos.y = xrOrigin.Camera.transform.position.y;
        teleportHandObject(RHinteractor, newPos);
        teleportHandObject(LHinteractor, newPos);
        mirrorElevator.tempDisabled = true;
        xrOrigin.MoveCameraToWorldLocation(newPos);
    }

    private IEnumerator ReenableSelfTeleport()
    {
        yield return new WaitForSeconds(1.5f);
        tempDisabled = false;
    }

    private void teleportHandObject(XRDirectInteractor interactor, Vector3 newPos)
    {
        if (interactor.hasSelection && interactor.selectTarget.TryGetComponent(out XRGrabInteractable grabInteractable))
        {
            GameObject grabbedObject = grabInteractable.gameObject;
            if (grabbedObject != null)
                grabbedObject.transform.position = newPos;
        }
    }
}