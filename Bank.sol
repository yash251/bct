//SPDX-License-Identifier: MIT
pragma solidity ^0.8.7;

contract BankAccount {
    address public owner;
    uint256 public balance; // public keyword creates a getter function for this variable. 3. returns balance

    constructor() {
        owner = msg.sender; // sets owner to the address of the person who deployed the contract
    }

    receive() external payable {
        // 1. deposit money function
        balance += msg.value;
    }

    function withdraw(uint256 amount, address payable destAddress) public {
        // 2. withdraw money function
        require(msg.sender == owner, "Only owner can withdraw");
        require(amount <= balance, "Insufficent funds");

        destAddress.transfer(amount); // transfer is inbuilt function
        balance -= amount;
    }
}
