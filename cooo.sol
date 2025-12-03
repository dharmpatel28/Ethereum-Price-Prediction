// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

contract AdviceFeeContract {
    address public owner;
    uint256 public fee;
    mapping(address => bool) public hasPaid;

    event PaymentReceived(address indexed user, uint256 amount);
    event PaymentStatus(address indexed user, bool hasPaidStatus); // New event
    event AdviceGiven(address indexed user, string advice);
    event Withdraw(address indexed owner, uint256 amount);

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    constructor(uint256 _feeInWei) {
        owner = msg.sender;
        fee = _feeInWei;
    }

    // Function to pay for advice
    function payForAdvice() external payable {
        require(msg.value == fee, "Incorrect ETH sent");
        hasPaid[msg.sender] = true; // Set the user's payment status to true
        emit PaymentReceived(msg.sender, msg.value);
        emit PaymentStatus(msg.sender, hasPaid[msg.sender]); // Emit the payment status
    }

    // Function to get advice after paying
    function getAdvice() external returns (string memory) {
        require(hasPaid[msg.sender], "You must pay first");
        
        uint256 random = uint256(
            keccak256(abi.encodePacked(block.timestamp, msg.sender))
        ) % 2;

        string memory advice = random == 0
            ? "Buy Ethereum"
            : "Sell Ethereum";

        hasPaid[msg.sender] = false; // Reset payment status
        emit AdviceGiven(msg.sender, advice); // Emit the advice given event
        return advice;
    }

    // Withdraw contract balance to owner
    function withdraw() external onlyOwner {
        uint256 balance = address(this).balance;
        require(balance > 0, "No balance");
        payable(owner).transfer(balance);
        emit Withdraw(owner, balance);
    }

    // Function to check contract balance
    function checkContractBalance() external view returns (uint256) {
        return address(this).balance;
    }

    // Function to check if a user has paid
    function checkHasPaid(address _user) external view returns (bool) {
        return hasPaid[_user];
    }
}
